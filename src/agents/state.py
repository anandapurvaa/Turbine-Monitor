from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SensorReading:
    engine_id: int
    rul: float
    sensors: Dict[str, float]


@dataclass
class WorkOrder:
    work_order_id: int
    engine_id: int
    priority: str
    description: str
    failure_mode_id: Optional[str]
    status: str


class AgentState(TypedDict):
    """State for the TurbineGuard multi-agent system."""
    
    # Input
    engine_id: int
    
    # Sensor data
    sensor_reading: Optional[SensorReading]
    
    # RUL prediction
    rul_prediction: Optional[float]
    rul_threshold: float  # e.g., 30 cycles
    
    # Anomaly detection
    anomaly_detected: bool
    anomaly_reason: Optional[str]
    
    # RCA investigation
    failure_mode_hypothesis: Optional[str]
    retrieved_manuals: List[Dict[str, Any]]
    
    # Dispatch
    work_order: Optional[WorkOrder]
    parts_check: Optional[Dict[str, Any]]
    
    # Execution control
    iteration_count: int
    max_iterations: int
    error_log: List[str]
    
    # Final output
    final_decision: Optional[str]