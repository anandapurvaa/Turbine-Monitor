# TurbineGuard – Predictive Maintenance for Turbofan Engines

TurbineGuard is an end‑to‑end predictive maintenance system for aircraft turbofan engines. It uses time‑series forecasting models to estimate the **Remaining Useful Life (RUL)** of engines based on sensor readings, and exposes the results via an interactive dashboard. The entire pipeline—data processing, model training, automated retraining, and serving—is deployed on **Google Cloud**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data](#data)
  - [Dataset](#dataset)
  - [Preprocessing](#preprocessing)
- [Model](#model)
  - [PatchTST](#patchtst)
  - [Training Objective](#training-objective)
- [Training & Evaluation](#training--evaluation)
  - [Local Training](#local-training)
  - [Metrics](#metrics)
- [Automated Retraining](#automated-retraining)
  - [Cloud Run Job](#cloud-run-job)
  - [Cloud Scheduler](#cloud-scheduler)
- [Dashboard](#dashboard)
  - [Features](#features)
  - [Deployment](#deployment)
- [Monitoring](#monitoring)
  - [Cloud Monitoring Dashboard](#cloud-monitoring-dashboard)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Data Processing](#data-processing)
  - [Training](#training)
  - [Retraining](#retraining)
  - [Dashboard](#dashboard-1)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## Overview

Modern aircraft engines generate large amounts of sensor data during operation. Predictive maintenance aims to use this data to:

- Estimate how much longer an engine can operate before failure (**RUL**).
- Schedule maintenance proactively, reducing unplanned downtime and costs.
- Improve safety by avoiding catastrophic failures.

TurbineGuard implements this workflow using:

- The **C-MAPSS** turbofan degradation simulation datasets (FD001–FD004).
- A **PatchTST** time‑series transformer model from Hugging Face `transformers`.
- Automated **retraining** via Cloud Run Jobs and Cloud Scheduler.
- A **Plotly Dash** dashboard deployed on Cloud Run for interactive exploration.

---

## Architecture

High‑level architecture on Google Cloud:

1. **Data Storage**
   - Raw and processed data stored in **Google Cloud Storage (GCS)** under `gs://turbineguard-data/`.
   - Processed splits: `train_FD00X_processed.csv`, `val_FD00X_processed.csv`, `test_FD00X_processed.csv`.

2. **Training & Retraining**
   - Local or CI/CD training using `train.py` and `retrain/retrain_model.py`.
   - Models saved as `best_model_FD00X_*.pt` in `gs://turbineguard-data/models/FD00X/`.

3. **Automated Retraining**
   - **Cloud Run Job**: runs the retraining script to completion, then exits.
   - **Cloud Scheduler**: triggers the job on a fixed cron schedule (e.g., nightly/weekly).

4. **Serving**
   - **Cloud Run Service** (`turbineguard-dash`) hosts a Plotly Dash app.
   - Dashboard reads processed data and latest models from GCS and displays:
     - RUL predictions vs. true RUL.
     - Error metrics (MAE, RMSE).
     - Sensor trends and degradation patterns.

5. **Monitoring**
   - **Cloud Monitoring Dashboard** tracks:
     - Cloud Run Job execution success/failure.
     - Cloud Scheduler last run time.

---

## Data

### Dataset

TurbineGuard uses the **C-MAPSS** (Commercial Modular Aero‑Propulsion System Simulation) datasets:

- **FD001**: Single operating condition, single fault mode.
- **FD002**: Multiple operating conditions, single fault mode.
- **FD003**: Single operating condition, multiple fault modes.
- **FD004**: Multiple operating conditions, multiple fault modes.

Each record contains:

- Engine ID
- Time step (cycle)
- Operational settings
- Multiple sensor readings
- Implicit RUL (derived from time to failure)

### Preprocessing

The preprocessing pipeline (`src/data_loader.py` and related scripts) performs:

- **RUL label creation**:  
  For each engine, RUL at time `t` is defined as `T_failure - t`.
- **Normalization**:  
  Sensors are normalized (e.g., min‑max or standard scaling) per dataset.
- **Sequence creation**:  
  Time series are split into fixed‑length windows (lookback windows) suitable for PatchTST.
- **Train/Val/Test splits**:  
  Per‑dataset splits saved as:
  - `train_FD00X_processed.csv`
  - `val_FD00X_processed.csv`
  - `test_FD00X_processed.csv`

These files are uploaded to GCS:

```bash
gcloud storage cp data/processed/*.csv gs://turbineguard-data/processed/
```

---

## Model

### PatchTST

TurbineGuard uses **PatchTST**, a transformer‑based time‑series model available via Hugging Face `transformers`:

- Inputs are split into **patches** (sub‑sequences) rather than individual time steps.
- A transformer encoder operates on patch embeddings, capturing long‑range dependencies efficiently.
- Well‑suited for multivariate sensor sequences with complex temporal patterns.

Implementation details:

- Model class: `PatchTSTModel` from `transformers`.
- Configuration: `PatchTSTConfig` (lookback window, patch size, number of layers, hidden size, etc.).
- Framework: PyTorch, trained with standard optimizers (e.g., AdamW) and learning rate schedules.

### Training Objective

The model is trained to **regress** the RUL:

- Input: sequence of sensor readings over a lookback window.
- Target: RUL at the last time step of the window.
- Loss: Mean Squared Error (MSE) or similar regression loss.

Evaluation metrics:

- **MAE (Mean Absolute Error)** – average absolute difference between predicted and true RUL.
- **RMSE (Root Mean Squared Error)** – penalizes larger errors more strongly.

---

## Training & Evaluation

### Local Training

Training is performed via `train.py` (or similar) using processed CSVs:

```bash
python -m train.train_model --dataset FD001 --epochs 50 --patience 5
```

Key steps:

1. Load processed CSVs from GCS or local disk.
2. Construct PyTorch datasets and dataloaders.
3. Initialize `PatchTSTModel` with `PatchTSTConfig`.
4. Train with early stopping based on validation loss.
5. Save the best checkpoint as `best_model_FD001.pt`.

### Metrics

For each dataset (FD001–FD004), the pipeline logs:

- Training loss per epoch.
- Validation loss per epoch.
- Final **MAE** and **RMSE** on validation and test sets.

Example log output:

```text
FD001 | Epoch 1/50 | Train: 5000.0 | Val: 2000.0 | MAE: 30.1 | RMSE: 38.4 | LR: 3.00e-04
...
FD001 | Best Val MAE: 14.2 | Test MAE: 13.8 | Test RMSE: 18.9
```

These metrics are used to compare configurations and track improvements over time.

---

## Automated Retraining

To keep models up‑to‑date and demonstrate MLOps best practices, TurbineGuard implements automated retraining on Google Cloud.

### Cloud Run Job

A dedicated container image (`turbineguard/retrain:latest`) runs the retraining script:

- Entrypoint: `python -m retrain.retrain_model`
- Behavior:
  - Downloads processed CSVs from `gs://turbineguard-data/processed/`.
  - Retrains models for FD001–FD004 (or a subset).
  - Uploads new checkpoints to `gs://turbineguard-data/models/FD00X/`.
  - Logs metrics to stdout (visible in Cloud Logging).

Deployment (example):

```bash
gcloud run jobs deploy turbineguard-retrain \
  --image europe-west3-docker.pkg.dev/cloudprojects-506123/turbineguard/retrain:latest \
  --region europe-west3 \
  --task-timeout 1h \
  --max-retries 1 \
  --set-env-vars GCS_BUCKET=turbineguard-data
```

Manual execution for testing:

```bash
gcloud run jobs execute turbineguard-retrain --region europe-west3 --wait
```

### Cloud Scheduler

A **Cloud Scheduler** job triggers the Cloud Run Job on a cron schedule (e.g., daily at 02:00 UTC):

- Target: Cloud Run Jobs `:run` endpoint:
  ```text
  https://run.googleapis.com/v2/projects/cloudprojects-506123/locations/europe-west3/jobs/turbineguard-retrain:run
  ```
- Auth: OIDC with a service account having `roles/run.invoker` on the job.

Example creation (if not already deployed):

```bash
gcloud scheduler jobs create http turbineguard-retrain-schedule \
  --location europe-west3 \
  --schedule "0 2 * * *" \
  --uri "https://run.googleapis.com/v2/projects/cloudprojects-506123/locations/europe-west3/jobs/turbineguard-retrain:run" \
  --http-method POST \
  --oauth-service-account-email "<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
  --time-zone "UTC"
```

This ensures models are periodically retrained without manual intervention.

---

## Dashboard

The dashboard provides an interactive interface for exploring RUL predictions and model performance.

### Features

- **RUL Prediction Plots**
  - True vs. predicted RUL over time for selected engines.
  - Separate views per dataset (FD001–FD004).

- **Error Analysis**
  - Distribution of prediction errors (MAE, RMSE).
  - Per‑engine and aggregate metrics.

- **Sensor Trends**
  - Time‑series plots of key sensors alongside RUL.
  - Helps interpret degradation patterns.

- **Model Info**
  - Displays which model checkpoint is currently loaded.
  - Shows high‑level training metrics (e.g., best MAE/RMSE).

Implementation:

- Built with **Plotly Dash**.
- Runs in a container deployed as a **Cloud Run Service** (`turbineguard-dash`).
- Listens on the `PORT` environment variable injected by Cloud Run.

### Deployment

CI/CD (GitHub Actions) builds and deploys the dashboard on every push to `main`:

- Dockerfile: `docker/Dockerfile.dash`
- Service: `turbineguard-dash` in `europe-west3`.
- Access: `--allow-unauthenticated` (can be restricted later).

Example manual deploy (if needed):

```bash
gcloud run deploy turbineguard-dash \
  --image europe-west3-docker.pkg.dev/cloudprojects-506123/turbineguard/dash-dashboard:latest \
  --platform managed \
  --region europe-west3 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

---

## Monitoring

To ensure reliability of the retraining pipeline, TurbineGuard uses **Cloud Monitoring**.

### Cloud Monitoring Dashboard

A custom dashboard (`TurbineGuard - Retraining & Scheduler`) tracks:

- **Cloud Run Job Execution Count**
  - Total completed task attempts, split by status (`succeeded` vs. `failed`).
  - Metric: `run.googleapis.com/job/completed_task_attempt_count`.

- **Cloud Scheduler Last Run Time**
  - Timestamp of the last execution of the scheduler job.
  - Metric: `scheduler.googleapis.com/last_job_run_time`.

Creation:

```bash
gcloud monitoring dashboards create \
  --config-from-file=monitoring-dashboard.json \
  --project=cloudprojects-506123
```

This dashboard is available under **Cloud Console → Monitoring → Dashboards** and provides at‑a‑glance visibility into the health of the retraining system.

---

## Project Structure

Typical repository layout:

```text
TurbineMonitor/
├─ dashboard/
│  └─ app.py                 # Dash app
├─ data/
│  ├─ raw/                   # Raw C-MAPSS files (optional)
│  └─ processed/             # Processed CSVs (train/val/test per FD00X)
├─ docker/
│  ├─ Dockerfile.dash        # Dashboard container
│  └─ Dockerfile.retrain     # Retraining job container
├─ retrain/
│  └─ retrain_model.py       # Automated retraining script
├─ src/
│  ├─ data_loader.py         # Data loading & preprocessing
│  └─ model.py               # PatchTST model wrapper
├─ train/
│  └─ train_model.py         # Training script
├─ .github/
│  └─ workflows/
│     ├─ deploy_dashboard.yml # CI/CD for dashboard
│     └─ deploy-retrain.yml   # CI/CD for retraining job
├─ monitoring-dashboard.json  # Cloud Monitoring dashboard config
└─ README.md
```

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- `pip` and virtualenv (or `venv`)
- Google Cloud SDK (`gcloud`)
- Docker (optional, for building images)
- Access to the C-MAPSS datasets

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

`requirements.txt` should include at least:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `dash`, `plotly`
- `google-cloud-storage`

Configure GCS access:

```bash
gcloud auth application-default login
gcloud config set project cloudprojects-506123
```

### Data Processing

Run your preprocessing script (exact command depends on your setup) to generate:

```text
data/processed/train_FD001_processed.csv
data/processed/val_FD001_processed.csv
data/processed/test_FD001_processed.csv
...
```

Upload to GCS (optional):

```bash
gcloud storage cp data/processed/*.csv gs://turbineguard-data/processed/
```

### Training

Example for FD001:

```bash
python -m train.train_model --dataset FD001 --epochs 50 --patience 5
```

Adjust arguments as needed (learning rate, lookback window, etc.).

### Retraining

To simulate the Cloud Run Job locally:

```bash
python -m retrain.retrain_model --dataset FD001 --epochs 50 --patience 5
```

Ensure `GCS_BUCKET` is set if reading/writing from GCS:

```powershell
$env:GCS_BUCKET = "turbineguard-data"
```

### Dashboard

From the project root:

```bash
cd dashboard
python app.py
```

Then open `http://127.0.0.1:8050` in your browser.

---

## Key Findings

Experiments with PatchTST on the C-MAPSS datasets yielded several insights:

### 1. Final Model Performance (context length = 56)

Using the final configuration (lookback window = 56, larger PatchTST encoder, SmoothL1 loss, and cosine LR schedule), the model achieved:

| Dataset | Test MAE | Test RMSE |
|---------|----------|-----------|
| FD001   | 11.38    | 14.60     |
| FD002   | 14.81    | 19.84     |
| FD003   | 11.76    | 16.50     |
| FD004   | 10.98    | 17.81     |

These results are competitive with recent deep learning baselines on C-MAPSS, especially considering the use of a single transformer model without heavy ensembling or extensive hand‑crafted features.

### 2. Strong Performance on FD001 and FD004

- **FD001** (single operating condition, single fault) showed low MAE and RMSE, with predicted RUL curves closely following true degradation trajectories.
- **FD004** (multiple operating conditions, multiple fault modes) achieved the **lowest MAE** (10.98) despite being the most complex dataset, indicating that the combination of regime‑aware normalization and a deeper PatchTST encoder effectively captures heterogeneous degradation patterns.

### 3. Balanced Accuracy Across All Datasets

- **FD002** (multiple operating conditions) and **FD003** (single condition, multiple faults) exhibited slightly higher MAE than FD001 but remained in a tight range (≈ 11–15), demonstrating robust generalization across different fault and operating regimes.
- The regime‑aware K‑Means normalization was particularly beneficial for FD002 and FD004, where distinct operating conditions would otherwise degrade performance.

### 4. Effect of Lookback Window and Model Capacity

- Increasing the context length from 36 to **56 cycles** allowed the model to observe longer degradation histories, improving RUL estimates especially for engines with gradual failure patterns.
- A larger encoder (d_model = 256, 4 layers, 8 attention heads) provided sufficient capacity to model complex temporal dependencies without overfitting, as evidenced by consistent validation and test metrics.

### 5. Benefits of Automated Retraining

- Periodic retraining ensures models adapt to new data or configuration changes.
- Cloud Run Jobs provide a simple, serverless way to run batch training.
- Cloud Scheduler guarantees regular execution without manual intervention.

### 6. End‑to‑End MLOps Pipeline

- The project demonstrates a complete MLOps workflow:
  - Data versioning in GCS.
  - Reproducible training scripts.
  - CI/CD for dashboard and retraining job deployment.
  - Scheduled retraining and monitoring.
- This architecture is directly applicable to real‑world predictive maintenance systems.

---

## Future Work

Potential extensions to further improve TurbineGuard:

- **Model Enhancements**
  - Experiment with different transformer architectures (e.g., Temporal Fusion Transformer).
  - Add uncertainty estimation (e.g., quantile regression, ensembles).

- **Feature Engineering**
  - Derive additional features (rolling statistics, health indices).
  - Incorporate operational settings more explicitly.

- **Advanced Monitoring & Alerting**
  - Alert on retraining failures or significant metric degradation.
  - Track data drift in sensor distributions over time.

- **User‑Facing Improvements**
  - Add "what‑if" scenarios (e.g., simulate maintenance actions).
  - Export predictions and reports for integration with maintenance systems.

---

**TurbineGuard** demonstrates that modern time‑series transformers, combined with cloud‑native infrastructure, can deliver accurate and operationally useful RUL predictions for turbofan engines.