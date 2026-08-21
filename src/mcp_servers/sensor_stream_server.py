from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("sensor-stream-server")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}
OP_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

_data_cache: dict[str, pd.DataFrame] = {}


def load_dataset(dataset: str) -> pd.DataFrame:
    dataset = dataset.upper()

    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Choose one of: {sorted(VALID_DATASETS)}"
        )

    if dataset not in _data_cache:
        path = DATA_DIR / f"test_{dataset}_processed.csv"

        if not path.exists():
            raise FileNotFoundError(f"Processed test data not found: {path}")

        _data_cache[dataset] = pd.read_csv(path)

    return _data_cache[dataset]


@mcp.tool()
def get_latest_readings(
    engine_id: int,
    dataset: str = "FD001",
    num_cycles: int = 36,
) -> dict:
    """
    Return the latest time-ordered cycles for one engine.

    Includes three operating settings and 21 sensors for each cycle, which
    provides the 24-channel PatchTST model input.
    """
    df = load_dataset(dataset)
    engine_data = (
        df[df["unit_nr"] == engine_id]
        .sort_values("time_in_cycles")
        .tail(num_cycles)
    )

    if engine_data.empty:
        return {
            "error": f"Engine {engine_id} not found in {dataset}",
            "available_engines": sorted(df["unit_nr"].unique().tolist()),
        }

    feature_columns = OP_COLUMNS + SENSOR_COLUMNS

    readings = []
    for _, row in engine_data.iterrows():
        readings.append(
            {
                "cycle": int(row["time_in_cycles"]),
                "operating_settings": {
                    column: float(row[column])
                    for column in OP_COLUMNS
                },
                "sensors": {
                    column: float(row[column])
                    for column in SENSOR_COLUMNS
                },
            }
        )

    latest = engine_data.iloc[-1]

    return {
        "dataset": dataset.upper(),
        "engine_id": int(engine_id),
        "num_cycles": int(len(engine_data)),
        "latest_cycle": int(latest["time_in_cycles"]),
        "feature_columns": feature_columns,
        "readings": readings,
        "latest_operating_settings": {
            column: float(latest[column])
            for column in OP_COLUMNS
        },
        "latest_sensors": {
            column: float(latest[column])
            for column in SENSOR_COLUMNS
        },
    }


@mcp.tool()
def list_engines(dataset: str = "FD001") -> list[int]:
    """List available engine IDs for a selected C-MAPSS dataset."""
    df = load_dataset(dataset)
    return sorted(df["unit_nr"].unique().tolist())


if __name__ == "__main__":
    mcp.run()