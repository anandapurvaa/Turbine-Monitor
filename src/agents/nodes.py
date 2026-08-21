from typing import Dict, Any, List
import torch
from transformers import PatchTSTModel, PatchTSTConfig
import numpy as np

from src.agents.state import AgentState, SensorReading, WorkOrder
from src.rag.index import ManualIndex
from src.agents.mcp_tools import create_work_order_mcp, check_parts_inventory_mcp
from src.mcp_servers.sensor_stream_server import get_latest_readings


# Load models and indices at module level (lazy loading)
_model = None
_model_config = None
_train_stats = None
_rag_indexer = None


def get_model_and_stats():
    global _model, _model_config, _train_stats
    if _model is None:
        checkpoint = torch.load("best_model.pt", map_location="cpu", weights_only=False)
        _model_config = PatchTSTConfig.from_dict(checkpoint["config"])
        _model = PatchTSTRegressor(_model_config)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        _train_stats = checkpoint["train_stats"]
    return _model, _model_config, _train_stats


def get_rag_indexer():
    global _rag_indexer
    if _rag_indexer is None:
        from pathlib import Path
        _rag_indexer = ManualIndex()
        _rag_indexer.load(Path("data/manuals/index"))
    return _rag_indexer


class PatchTSTRegressor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = PatchTSTModel(config)
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.head_dropout),
            torch.nn.Linear(config.d_model // 2, 1)
        )
        
    def forward(self, x):
        outputs = self.encoder(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=(1, 2))
        pred = self.regression_head(pooled).squeeze(-1)
        return pred
    

def anomaly_detector_node(state: AgentState) -> AgentState:
    """
    Anomaly Detector: Reads sensor stream and predicts RUL.
    Flags anomaly if RUL breaches threshold.
    """
    engine_id = state["engine_id"]
    
    # Get latest sensor readings
    reading_data = get_latest_readings(engine_id, num_cycles=36)
    
    if "error" in reading_data:
        state["error_log"].append(f"Sensor read error: {reading_data['error']}")
        state["iteration_count"] += 1
        state["final_decision"] = "Sensor read error"
        return state
    
    # For demo, use the current RUL from the data as prediction
    predicted_rul = reading_data["current_rul"]
    
    state["sensor_reading"] = SensorReading(
        engine_id=engine_id,
        rul=predicted_rul,
        sensors=reading_data["sensors"]
    )
    state["rul_prediction"] = predicted_rul
    
    # Check threshold
    threshold = state["rul_threshold"]
    if predicted_rul < threshold:
        state["anomaly_detected"] = True
        state["anomaly_reason"] = f"RUL {predicted_rul:.1f} cycles below threshold {threshold}"
    else:
        state["anomaly_detected"] = False
        state["anomaly_reason"] = f"RUL {predicted_rul:.1f} cycles above threshold {threshold}"
        # Set final decision for healthy engines
        state["final_decision"] = f"Engine {engine_id} healthy: {state['anomaly_reason']}"
    
    state["iteration_count"] += 1
    return state


def rca_investigator_node(state: AgentState) -> AgentState:
    """
    RCA Investigator: Searches maintenance manuals for failure mode diagnosis.
    """
    if not state["anomaly_detected"]:
        state["iteration_count"] += 1
        return state
    
    # Build query from sensor patterns
    reading = state["sensor_reading"]
    sensors = reading.sensors
    
    # Identify anomalous sensors (top 3 deviations from typical operating range)
    # Simplified heuristic for demo
    sensor_deviations = {
        k: abs(v - 0.5) for k, v in sensors.items()
    }
    top_sensors = sorted(sensor_deviations.items(), key=lambda x: x[1], reverse=True)[:3]
    
    query = f"Engine degradation with anomalies in {', '.join([s[0] for s in top_sensors])}"
    
    # Search manuals
    indexer = get_rag_indexer()
    results = indexer.search(query, top_k=2)
    
    state["retrieved_manuals"] = [
        {"id": m["id"], "title": m["title"], "score": s}
        for m, s in results
    ]
    
    if results:
        top_manual = results[0][0]
        state["failure_mode_hypothesis"] = top_manual["title"]
    else:
        state["failure_mode_hypothesis"] = "Unknown failure mode"
    
    state["iteration_count"] += 1
    return state


def dispatcher_node(state: AgentState) -> AgentState:
    """
    Dispatcher: Creates work order and checks parts inventory.
    """
    if not state["anomaly_detected"]:
        state["iteration_count"] += 1
        return state
    
    engine_id = state["engine_id"]
    rul = state["rul_prediction"]
    failure_mode = state["failure_mode_hypothesis"]
    
    # Determine priority
    if rul < 15:
        priority = "HIGH"
    elif rul < 30:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    # Create work order
    description = f"Engine {engine_id} RUL {rul:.1f} cycles. Suspected: {failure_mode}"
    work_order_data = create_work_order_mcp(
        engine_id=engine_id,
        priority=priority,
        description=description,
        failure_mode_id=failure_mode
    )
    
    state["work_order"] = WorkOrder(
        work_order_id=work_order_data["work_order_id"],
        engine_id=engine_id,
        priority=priority,
        description=description,
        failure_mode_id=failure_mode,
        status=work_order_data["status"]
    )
    
    # Check parts inventory (example: check HPC blade for HPC degradation)
    if "HPC" in failure_mode:
        parts_check = check_parts_inventory_mcp("HPC-BLADE-001")
        state["parts_check"] = parts_check
    
    state["iteration_count"] += 1
    return state


def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor: Decides next action based on current state.
    """
    iteration = state["iteration_count"]
    max_iter = state["max_iterations"]
    
    # Check iteration limit
    if iteration >= max_iter:
        state["final_decision"] = f"Max iterations ({max_iter}) reached"
        return state
    
    # Decision logic
    if not state["anomaly_detected"]:
        state["final_decision"] = f"No anomaly: {state['anomaly_reason']}"
        return state
    
    if state["work_order"] is not None:
        state["final_decision"] = (
            f"Work order #{state['work_order'].work_order_id} created. "
            f"Priority: {state['work_order'].priority}. "
            f"Failure mode: {state['failure_mode_hypothesis']}"
        )
        return state
    
    # Continue workflow
    return state


def should_continue(state: AgentState) -> str:
    """Conditional edge: decide which node to run next."""
    # If final decision already set, end
    if state["final_decision"] is not None:
        return "END"
    
    # If no anomaly and we've processed, end
    if not state["anomaly_detected"] and state["anomaly_reason"] is not None:
        return "END"
    
    # Workflow: anomaly_detector -> rca_investigator -> dispatcher
    if state["sensor_reading"] is None:
        return "anomaly_detector"
    elif state["failure_mode_hypothesis"] is None and state["anomaly_detected"]:
        return "rca_investigator"
    elif state["work_order"] is None and state["anomaly_detected"]:
        return "dispatcher"
    else:
        return "supervisor"