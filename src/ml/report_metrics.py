import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def nasa_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    errors = predictions - targets
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(penalties.sum())


def metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "MAE": mae(predictions, targets),
        "RMSE": rmse(predictions, targets),
        "NASA Score": nasa_score(predictions, targets),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Report window-level and final-window-per-engine RUL metrics."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/metrics_summary.csv"),
    )
    args = parser.parse_args()

    required_columns = {
        "unit_number",
        "cycle",
        "predicted_rul",
        "actual_rul",
    }
    rows = []

    for dataset in ["FD001", "FD002", "FD003", "FD004"]:
        path = args.predictions_dir / f"predictions_{dataset}.csv"

        if not path.exists():
            print(f"Skipping {dataset}: file not found: {path}")
            continue

        df = pd.read_csv(path)
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"{path} uses the old inference format and is missing: {sorted(missing)}. "
                "Run inference again using the updated inference.py file."
            )

        df = df.dropna(subset=list(required_columns)).copy()
        if df.empty:
            raise ValueError(f"{path} has no valid prediction rows.")

        window_metrics = metrics(
            df["predicted_rul"].to_numpy(),
            df["actual_rul"].to_numpy(),
        )

        final_by_engine = (
            df.sort_values(["unit_number", "cycle"])
            .groupby("unit_number", as_index=False)
            .tail(1)
        )
        engine_metrics = metrics(
            final_by_engine["predicted_rul"].to_numpy(),
            final_by_engine["actual_rul"].to_numpy(),
        )

        rows.append(
            {
                "Dataset": dataset,
                "Windows": len(df),
                "Engines": final_by_engine["unit_number"].nunique(),
                "Window MAE": window_metrics["MAE"],
                "Window RMSE": window_metrics["RMSE"],
                "Window NASA Score": window_metrics["NASA Score"],
                "Final-window MAE": engine_metrics["MAE"],
                "Final-window RMSE": engine_metrics["RMSE"],
                "Final-window NASA Score": engine_metrics["NASA Score"],
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No prediction files were found in {args.predictions_dir}."
        )

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    formatted = summary.copy()
    metric_columns = [
        "Window MAE",
        "Window RMSE",
        "Window NASA Score",
        "Final-window MAE",
        "Final-window RMSE",
        "Final-window NASA Score",
    ]
    for column in metric_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")

    print("\nRUL evaluation summary")
    print("=" * 120)
    print(formatted.to_string(index=False))
    print("=" * 120)
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()