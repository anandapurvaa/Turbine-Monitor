from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import (
    anomaly_detector_node,
    rca_investigator_node,
    dispatcher_node,
    supervisor_node,
    should_continue,
)


def build_turbineguard_graph():
    """Build the TurbineGuard multi-agent workflow graph."""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("anomaly_detector", anomaly_detector_node)
    workflow.add_node("rca_investigator", rca_investigator_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("supervisor", supervisor_node)
    
    # Set entry point
    workflow.set_entry_point("anomaly_detector")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "anomaly_detector",
        should_continue,
        {
            "anomaly_detector": "anomaly_detector",
            "rca_investigator": "rca_investigator",
            "dispatcher": "dispatcher",
            "supervisor": "supervisor",
            "END": END,
        }
    )
    
    workflow.add_conditional_edges(
        "rca_investigator",
        should_continue,
        {
            "rca_investigator": "rca_investigator",
            "dispatcher": "dispatcher",
            "supervisor": "supervisor",
            "END": END,
        }
    )
    
    workflow.add_conditional_edges(
        "dispatcher",
        should_continue,
        {
            "dispatcher": "dispatcher",
            "supervisor": "supervisor",
            "END": END,
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "supervisor": "supervisor",
            "END": END,
        }
    )
    
    # Compile
    app = workflow.compile()
    return app


def run_turbineguard(engine_id: int, rul_threshold: float = 30.0):
    """Run the full TurbineGuard workflow for a given engine."""
    
    app = build_turbineguard_graph()
    
    # Initial state
    initial_state = {
        "engine_id": engine_id,
        "sensor_reading": None,
        "rul_prediction": None,
        "rul_threshold": rul_threshold,
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
    
    # Run
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Test run
    result = run_turbineguard(engine_id=1, rul_threshold=30.0)
    
    print("\n=== TurbineGuard Decision ===")
    print(result["final_decision"])
    
    if result.get("work_order"):
        wo = result["work_order"]
        print(f"\nWork Order #{wo.work_order_id}")
        print(f"  Engine: {wo.engine_id}")
        print(f"  Priority: {wo.priority}")
        print(f"  Failure Mode: {wo.failure_mode_id}")
    
    if result.get("retrieved_manuals"):
        print("\nRetrieved Manuals:")
        for manual in result["retrieved_manuals"]:
            print(f"  - {manual['title']} (score: {manual['score']:.3f})")