from pathlib import Path
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np

# Initialize FastMCP server
mcp = FastMCP("sensor-stream-server")

# Load test data for mock streaming
TEST_DATA_PATH = Path("data/processed/test_FD001_processed.csv")
test_data = pd.read_csv(TEST_DATA_PATH)


@mcp.tool()
def get_latest_readings(engine_id: int, num_cycles: int = 10) -> dict:
    """
    Get recent sensor readings for a specific engine.
    
    Args:
        engine_id: Engine unit number
        num_cycles: Number of recent cycles to return (default: 10)
    
    Returns:
        Sensor readings with metadata
    """
    engine_data = test_data[test_data["unit_nr"] == engine_id].tail(num_cycles)
    
    if engine_data.empty:
        return {
            "error": f"Engine {engine_id} not found",
            "available_engines": sorted(test_data["unit_nr"].unique().tolist())
        }
    
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    
    return {
        "engine_id": engine_id,
        "num_cycles": len(engine_data),
        "latest_cycle": int(engine_data["time_in_cycles"].iloc[-1]),
        "current_rul": float(engine_data["RUL_capped"].iloc[-1]),
        "sensors": {
            col: float(engine_data[col].iloc[-1])
            for col in sensor_cols
        },
        "recent_trend": {
            col: float(engine_data[col].iloc[-1] - engine_data[col].iloc[0])
            for col in sensor_cols
        }
    }


@mcp.tool()
def list_engines() -> list[int]:
    """
    List all available engine IDs in the system.
    
    Returns:
        Sorted list of engine unit numbers
    """
    return sorted(test_data["unit_nr"].unique().tolist())


if __name__ == "__main__":
    mcp.run()