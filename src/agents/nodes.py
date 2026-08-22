from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.agents.state import AgentState, SensorReading, WorkOrder
from src.rag.index import ManualIndex
from src.agents.mcp_tools import check_parts_inventory_mcp
from src.ml.dashboard_inference import predict_engine_rul


_rag_indexer = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

OP_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def get_rag_indexer():
    global _rag_indexer

    if _rag_indexer is None:
        _rag_indexer = ManualIndex()
        _rag_indexer.load(PROJECT_ROOT / "data" / "manuals" / "index")

    return _rag_indexer


def load_engine_window(dataset: str, engine_id: int) -> pd.DataFrame:
    """Load available test-history for one engine, sorted by cycle."""
    dataset = dataset.upper()
    path = DATA_DIR / f"test_{dataset}_processed.csv"

    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")

    df = pd.read_csv(path)
    engine_window = (
        df[df["unit_nr"] == engine_id]
        .sort_values("time_in_cycles")
        .copy()
    )

    if engine_window.empty:
        available = sorted(df["unit_nr"].unique().tolist())
        raise ValueError(
            f"Engine {engine_id} was not found in {dataset}. "
            f"Available engines include: {available[:10]}"
        )

    return engine_window


def anomaly_detector_node(state: AgentState) -> AgentState:
    """
    Run real PatchTST model inference for the selected dataset and engine.

    This does not use RUL_capped as a prediction. RUL_capped remains only in
    the local evaluation data and is not passed to the model.
    """
    engine_id = state["engine_id"]
    dataset = state["dataset"]

    try:
        engine_window = load_engine_window(dataset, engine_id)
        prediction = predict_engine_rul(dataset, engine_window)

        predicted_rul = float(prediction["predicted_rul"])
        operating_regime = int(prediction["operating_regime"])
        latest_cycle = int(engine_window["time_in_cycles"].iloc[-1])
        latest_row = engine_window.iloc[-1]

        latest_sensors: Dict[str, float] = {
            sensor: float(latest_row[sensor])
            for sensor in SENSOR_COLUMNS
        }

        state["sensor_reading"] = SensorReading(
            engine_id=engine_id,
            rul=predicted_rul,
            sensors=latest_sensors,
        )
        state["rul_prediction"] = predicted_rul
        state["operating_regime"] = operating_regime
        state["latest_cycle"] = latest_cycle

        threshold = state["rul_threshold"]

        if predicted_rul < threshold:
            state["anomaly_detected"] = True
            state["anomaly_reason"] = (
                f"Model-predicted RUL {predicted_rul:.1f} cycles is below "
                f"the {threshold:.0f}-cycle threshold."
            )
        else:
            state["anomaly_detected"] = False
            state["anomaly_reason"] = (
                f"Model-predicted RUL {predicted_rul:.1f} cycles is above "
                f"the {threshold:.0f}-cycle threshold."
            )
            state["final_decision"] = (
                f"Engine {engine_id} ({dataset}) is healthy. "
                f"{state['anomaly_reason']}"
            )

    except Exception as exc:
        state["error_log"].append(str(exc))
        state["final_decision"] = f"Prediction failed: {exc}"

    state["iteration_count"] += 1
    return state


def rca_investigator_node(state: AgentState) -> AgentState:
    """Generate a cautious, sensor-trend-based failure hypothesis and retrieve manuals."""
    if not state["anomaly_detected"]:
        state["iteration_count"] += 1
        return state

    engine_id = state["engine_id"]
    dataset = state["dataset"]

    try:
        engine_window = load_engine_window(dataset, engine_id)

        # Compare the recent operating window with earlier engine history.
        recent_n = min(20, max(5, len(engine_window) // 4))
        baseline_end = max(recent_n, len(engine_window) - recent_n)

        recent = engine_window.tail(recent_n)
        baseline = engine_window.iloc[:baseline_end]

        sensor_scores = {}

        for sensor in SENSOR_COLUMNS:
            baseline_values = baseline[sensor].to_numpy(dtype=np.float32)
            recent_values = recent[sensor].to_numpy(dtype=np.float32)

            baseline_mean = float(np.mean(baseline_values))
            baseline_std = float(np.std(baseline_values))

            # Prevent division by zero for nearly constant sensor channels.
            if baseline_std < 1e-6:
                baseline_std = 1.0

            recent_mean = float(np.mean(recent_values))
            trend = float(recent_values[-1] - recent_values[0])

            # Normalized recent-vs-baseline deviation plus normalized local trend.
            shift_score = abs(recent_mean - baseline_mean) / baseline_std
            trend_score = abs(trend) / baseline_std

            sensor_scores[sensor] = shift_score + 0.5 * trend_score

        top_sensors = sorted(
            sensor_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        top_sensor_names = [name for name, _ in top_sensors]
        evidence = ", ".join(
            f"{name} (anomaly score {score:.2f})"
            for name, score in top_sensors
        )

        # C-MAPSS sensor numbers alone do not prove a physical fault.
        # These are deliberately phrased as cautious hypotheses.
        if any(sensor in {"sensor_2", "sensor_3", "sensor_4", "sensor_11"} for sensor in top_sensor_names):
            hypothesis = "Suspected core-temperature or turbine-efficiency degradation"
            query = (
                "turbofan maintenance manual core temperature turbine efficiency "
                f"degradation evidence {', '.join(top_sensor_names)}"
            )
        elif any(sensor in {"sensor_7", "sensor_8", "sensor_9"} for sensor in top_sensor_names):
            hypothesis = "Suspected compressor pressure-performance degradation"
            query = (
                "turbofan maintenance manual compressor pressure performance "
                f"degradation evidence {', '.join(top_sensor_names)}"
            )
        elif any(sensor in {"sensor_12", "sensor_13", "sensor_14"} for sensor in top_sensor_names):
            hypothesis = "Suspected fuel-flow or bypass-flow anomaly"
            query = (
                "turbofan maintenance manual fuel flow bypass flow anomaly "
                f"evidence {', '.join(top_sensor_names)}"
            )
        elif any(sensor in {"sensor_15", "sensor_16", "sensor_17", "sensor_20", "sensor_21"} for sensor in top_sensor_names):
            hypothesis = "Suspected rotational-speed or cooling-system degradation"
            query = (
                "turbofan maintenance manual rotational speed cooling system "
                f"degradation evidence {', '.join(top_sensor_names)}"
            )
        else:
            hypothesis = "Suspected multivariate engine-performance degradation"
            query = (
                "turbofan maintenance manual multivariate engine degradation "
                f"evidence {', '.join(top_sensor_names)}"
            )

        indexer = get_rag_indexer()
        results = indexer.search(query, top_k=2)

        state["retrieved_manuals"] = [
            {
                "id": manual["id"],
                "title": manual["title"],
                "score": float(score),
            }
            for manual, score in results
        ]

        state["failure_mode_hypothesis"] = (
            f"{hypothesis}. Evidence: {evidence}."
        )

    except Exception as exc:
        state["error_log"].append(f"RCA lookup failed: {exc}")
        state["retrieved_manuals"] = []
        state["failure_mode_hypothesis"] = (
            "Suspected general engine degradation; sensor-trend analysis was unavailable."
        )

    state["iteration_count"] += 1
    return state


def dispatcher_node(state: AgentState) -> AgentState:
    """
    Create a proposed work order only.

    No MCP create_work_order call occurs here. The dashboard is safe to use
    without creating external or persistent CMMS records.
    """
    if not state["anomaly_detected"]:
        state["iteration_count"] += 1
        return state

    engine_id = state["engine_id"]
    dataset = state["dataset"]
    rul = float(state["rul_prediction"])
    failure_mode = state["failure_mode_hypothesis"] or "General engine degradation"

    if rul < 15:
        priority = "HIGH"
    elif rul < 30:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    description = (
        f"PROPOSED WORK ORDER — {dataset} engine {engine_id}. "
        f"PatchTST predicted RUL: {rul:.1f} cycles. "
        f"Suspected condition: {failure_mode}."
    )

    state["work_order"] = WorkOrder(
        work_order_id=0,
        engine_id=engine_id,
        priority=priority,
        description=description,
        failure_mode_id=failure_mode,
        status="PROPOSED",
    )

    if "HPC" in failure_mode.upper():
        try:
            state["parts_check"] = check_parts_inventory_mcp("HPC-BLADE-001")
        except Exception as exc:
            state["error_log"].append(f"Parts check failed: {exc}")

    state["iteration_count"] += 1
    return state


def supervisor_node(state: AgentState) -> AgentState:
    """Produce the final agent decision."""
    if state["final_decision"] is not None:
        return state

    if not state["anomaly_detected"]:
        state["final_decision"] = (
            f"No anomaly detected. {state['anomaly_reason']}"
        )
        return state

    work_order = state["work_order"]

    if work_order is not None:
        state["final_decision"] = (
            f"Maintenance action recommended for {state['dataset']} engine "
            f"{state['engine_id']}. Proposed priority: {work_order.priority}. "
            f"Predicted RUL: {state['rul_prediction']:.1f} cycles."
        )
    else:
        state["final_decision"] = (
            f"Anomaly detected: {state['anomaly_reason']}"
        )

    return state


def should_continue(state: AgentState) -> str:
    """Choose the next LangGraph node."""
    if state["final_decision"] is not None:
        return "END"

    if state["sensor_reading"] is None:
        return "anomaly_detector"

    if state["anomaly_detected"] and state["failure_mode_hypothesis"] is None:
        return "rca_investigator"

    if state["anomaly_detected"] and state["work_order"] is None:
        return "dispatcher"

    return "supervisor"