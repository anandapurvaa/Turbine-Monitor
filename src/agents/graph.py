from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    anomaly_detector_node,
    dispatcher_node,
    rca_investigator_node,
    should_continue,
    supervisor_node,
)
from src.agents.state import AgentState


def build_turbineguard_graph():
    """Build and compile the TurbineGuard multi-agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("anomaly_detector", anomaly_detector_node)
    workflow.add_node("rca_investigator", rca_investigator_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("supervisor", supervisor_node)

    workflow.set_entry_point("anomaly_detector")

    workflow.add_conditional_edges(
        "anomaly_detector",
        should_continue,
        {
            "anomaly_detector": "anomaly_detector",
            "rca_investigator": "rca_investigator",
            "dispatcher": "dispatcher",
            "supervisor": "supervisor",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "rca_investigator",
        should_continue,
        {
            "dispatcher": "dispatcher",
            "supervisor": "supervisor",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "dispatcher",
        should_continue,
        {
            "supervisor": "supervisor",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "END": END,
        },
    )

    return workflow.compile()


def run_turbineguard(
    engine_id: int,
    dataset: str = "FD001",
    rul_threshold: float = 30.0,
):
    """Run real PatchTST inference and the TurbineGuard workflow."""
    app = build_turbineguard_graph()

    initial_state: AgentState = {
        "engine_id": int(engine_id),
        "dataset": dataset.upper(),
        "sensor_reading": None,
        "operating_regime": None,
        "latest_cycle": None,
        "rul_prediction": None,
        "rul_threshold": float(rul_threshold),
        "anomaly_detected": False,
        "anomaly_reason": None,
        "failure_mode_hypothesis": None,
        "retrieved_manuals": [],
        "work_order": None,
        "parts_check": None,
        "iteration_count": 0,
        "max_iterations": 10,
        "error_log": [],
        "final_decision": None,
    }

    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_turbineguard(
        engine_id=1,
        dataset="FD004",
        rul_threshold=30.0,
    )

    print("\n=== TurbineGuard Decision ===")
    print(result["final_decision"])