# TurbineGuard System Architecture

This document describes the end‑to‑end architecture of TurbineGuard, from data ingestion to model serving and automated retraining.

---

## High‑Level Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Google Cloud Platform                        │
│                                                                       │
│  ┌──────────────┐        ┌───────────────────────────────────────┐  │
│  │  C‑MAPSS     │        │         Google Cloud Storage          │  │
│  │  Datasets    │ ─────► │  gs://turbineguard-data/              │  │
│  │  (local)     │        │  - processed/                         │  │
│  └──────────────┘        │  - models/FD001, FD002, FD003, FD004  │  │
│                          └───────────────────────────────────────┘  │
│                                       │                             │
│                                       │                             │
│                    ┌──────────────────┴──────────────────┐         │
│                    │                                     │         │
│           ┌────────▼─────────┐                ┌──────────▼───────┐ │
│           │  Cloud Run Job   │                │  Cloud Run       │ │
│           │  (retraining)    │                │  Service         │ │
│           │  - turbineguard- │                │  - turbineguard- │ │
│           │    retrain       │                │    dash          │ │
│           │  - scheduled by  │                │  - Plotly Dash   │ │
│           │    Cloud Scheduler│               │    UI            │ │
│           └──────────────────┘                └──────────────────┘ │
│                    │                                     ▲         │
│                    │                                     │         │
│                    └──────────────────┬──────────────────┘         │
│                                       │                             │
│                          ┌────────────▼────────────┐               │
│                          │  Cloud Monitoring       │               │
│                          │  - Job success/failure  │               │
│                          │  - Scheduler last run   │               │
│                          └─────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Data Layer – Google Cloud Storage (GCS)

**Bucket:** `gs://turbineguard-data`

**Contents:**

- `processed/`  
  - `train_FD00X_processed.csv`  
  - `val_FD00X_processed.csv`  
  - `test_FD00X_processed.csv`  
  For X ∈ {001, 002, 003, 004}.

- `models/FD00X/`  
  - `best_model_FD00X_YYYYMMDD_HHMMSS.pt`  
  Checkpoints saved by training/retraining jobs.

**Purpose:**

- Central, versioned storage for all datasets and model checkpoints.
- Accessed by:
  - Local training scripts.
  - Cloud Run retraining job.
  - Dashboard inference code.

---

### 2. Training & Retraining

#### Local Training

- Scripts: `train/train_model.py`, `src/ml/dataset.py`.
- Runs on a local machine or VM with GPU/CPU.
- Reads processed CSVs from `data/processed/` (or directly from GCS).
- Saves best checkpoints as `best_model_FD00X.pt` locally and/or to GCS.

#### Automated Retraining – Cloud Run Job

**Job name:** `turbineguard-retrain`  
**Region:** `europe-west3`  
**Image:** `europe-west3-docker.pkg.dev/cloudprojects-506123/turbineguard/retrain:latest`

**Behavior:**

1. On each execution:
   - Downloads `processed/*.csv` from GCS to a temporary directory.
   - For each dataset (FD001–FD004):
     - Trains a PatchTST model (context length = 56).
     - Saves the best checkpoint (`best_model_FD00X.pt`) based on validation MAE.
   - Uploads new checkpoints to `gs://turbineguard-data/models/FD00X/`.
   - Logs metrics to MLflow and Cloud Logging.

2. Runs to completion and exits (batch job pattern).

**CI/CD:**

- GitHub Actions workflow: `.github/workflows/deploy-retrain.yml`.
- On push to `main` (with changes to `retrain/`, `src/`, or `Dockerfile.retrain`):
  - Builds and pushes the retrain image.
  - Updates the Cloud Run Job with the new image.

---

### 3. Scheduling – Cloud Scheduler

**Job name:** `turbineguard-retrain-schedule` (or similar)  
**Region:** `europe-west3`  
**Schedule:** Cron expression (e.g., `0 2 * * *` for daily at 02:00 UTC).

**Target:**

- Cloud Run Jobs `:run` endpoint:
  ```text
  https://run.googleapis.com/v2/projects/cloudprojects-506123/locations/europe-west3/jobs/turbineguard-retrain:run
  ```

**Auth:**

- OIDC with the project’s default compute service account.
- Requires `roles/run.invoker` on the Cloud Run Job.

**Purpose:**

- Automatically triggers retraining on a fixed schedule.
- Ensures models stay up‑to‑date without manual intervention.

---

### 4. Serving – Cloud Run Service (Dashboard)

**Service name:** `turbineguard-dash`  
**Region:** `europe-west3`  
**Image:** `europe-west3-docker.pkg.dev/cloudprojects-506123/turbineguard/dash-dashboard:latest`

**Implementation:**

- Plotly Dash application (`dashboard/app.py`).
- Inference logic in `src/ml/dashboard_inference.py`:
  - Loads `best_model_FD00X.pt` from the local filesystem (mounted or baked into the image).
  - Uses regime‑aware normalization statistics stored in the checkpoint.
  - Predicts RUL for a selected engine using the most recent `context_length` cycles.

**Features:**

- Interactive selection of:
  - Dataset (FD001–FD004).
  - Engine ID.
  - RUL threshold for risk classification.
- Displays:
  - Predicted RUL and current cycle.
  - Operating regime.
  - Agent decision banner (risk status + narrative).
  - Proposed work order (simulated).
  - Retrieved maintenance manuals (RAG via FAISS).
  - Fleet‑wide RUL distribution chart.

**CI/CD:**

- GitHub Actions workflow: `.github/workflows/deploy_dashboard.yml`.
- On push to `main`:
  - Builds and pushes the dashboard image.
  - Deploys a new revision of the Cloud Run Service.

---

### 5. Monitoring – Cloud Monitoring Dashboard

**Dashboard name:** `TurbineGuard - Retraining & Scheduler`

**Key charts:**

- **Cloud Run Job Execution Count**
  - Metric: `run.googleapis.com/job/completed_task_attempt_count`
  - Grouped by `status` (succeeded/failed).
  - Shows how many retraining runs completed and whether they succeeded.

- **Cloud Scheduler Last Run Time**
  - Metric: `scheduler.googleapis.com/last_job_run_time`
  - Indicates when the scheduler last triggered the retraining job.

**Purpose:**

- Operational visibility into the retraining pipeline.
- Quick detection of:
  - Failed retraining runs.
  - Scheduler misconfigurations or missed executions.

---

## Data Flow Summary

1. **Offline / Batch**
   - Raw C-MAPSS data → preprocessing → `gs://turbineguard-data/processed/`.
   - Training scripts (local or Cloud Run Job) → trained models → `gs://turbineguard-data/models/`.

2. **Scheduled Retraining**
   - Cloud Scheduler → triggers Cloud Run Job `turbineguard-retrain`.
   - Job reads `processed/` from GCS, retrains models, uploads new checkpoints.

3. **Online Inference (Dashboard)**
   - User opens dashboard URL.
   - Dashboard loads latest checkpoints and processed test data.
   - For a selected engine:
     - Constructs the most recent window of length `context_length`.
     - Normalizes using regime‑aware statistics.
     - Runs PatchTST to predict RUL.
   - Displays predictions, risk status, and agent‑generated maintenance recommendations.

---

## Security & Permissions (Summary)

- **GCS access:**
  - Default compute service account has:
    - `roles/storage.objectViewer`
    - `roles/storage.objectCreator`
  - Used by both retraining job and dashboard (if reading from GCS).

- **Cloud Run Job:**
  - Runs as default compute service account (no explicit `--service-account`).
  - Invoked by Cloud Scheduler via OIDC.

- **Cloud Run Service:**
  - Deployed with `--allow-unauthenticated` for demo purposes (can be restricted to specific users/domains in production).

---

This architecture provides a clean separation between:

- **Data & model storage** (GCS),
- **Batch training/retraining** (Cloud Run Jobs + Scheduler),
- **Interactive serving** (Cloud Run Service),
- **Operational monitoring** (Cloud Monitoring),

forming a complete, cloud‑native predictive maintenance system.