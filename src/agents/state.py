from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


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

    engine_id: int
    dataset: str

    sensor_reading: Optional[SensorReading]
    operating_regime: Optional[int]
    latest_cycle: Optional[int]

    rul_prediction: Optional[float]
    rul_threshold: float

    anomaly_detected: bool
    anomaly_reason: Optional[str]

    failure_mode_hypothesis: Optional[str]
    retrieved_manuals: List[Dict[str, Any]]

    work_order: Optional[WorkOrder]
    parts_check: Optional[Dict[str, Any]]

    iteration_count: int
    max_iterations: int
    error_log: List[str]

    final_decision: Optional[str]