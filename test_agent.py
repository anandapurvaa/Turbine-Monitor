from src.agents.graph import run_turbineguard

# Test with engine 34 which has RUL = 7 (very low!)
result = run_turbineguard(engine_id=34, rul_threshold=30.0)

print("=== TurbineGuard Decision ===")
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

if result.get("parts_check"):
    print("\nParts Check:")
    print(f"  {result['parts_check']}")