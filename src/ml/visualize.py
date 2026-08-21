import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, fd in enumerate(["FD001", "FD002", "FD003", "FD004"]):
        pred_file = args.predictions_dir / f"predictions_{fd}.csv"
        if not pred_file.exists():
            print(f"⚠️  Skipping {fd} - no predictions found")
            continue

        df = pd.read_csv(pred_file)
        
        ax = axes[idx]
        ax.scatter(df["actual_rul"], df["predicted_rul"], alpha=0.3, s=10)
        
        # Perfect prediction line
        max_rul = max(df["actual_rul"].max(), df["predicted_rul"].max())
        ax.plot([0, max_rul], [0, max_rul], "r--", linewidth=2, label="Perfect")
        
        ax.set_xlabel("Actual RUL", fontsize=11)
        ax.set_ylabel("Predicted RUL", fontsize=11)
        ax.set_title(f"{fd} - Predicted vs Actual RUL", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        mae = np.mean(np.abs(df["predicted_rul"] - df["actual_rul"]))
        rmse = np.sqrt(np.mean((df["predicted_rul"] - df["actual_rul"]) ** 2))
        ax.text(0.05, 0.95, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}", 
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.output_dir / "predictions_all.png", dpi=150, bbox_inches="tight")
    print(f"Saved plots to {args.output_dir / 'predictions_all.png'}")

    # Individual plots
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        pred_file = args.predictions_dir / f"predictions_{fd}.csv"
        if not pred_file.exists():
            continue

        df = pd.read_csv(pred_file)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df["actual_rul"], df["predicted_rul"], alpha=0.3, s=10)
        
        max_rul = max(df["actual_rul"].max(), df["predicted_rul"].max())
        plt.plot([0, max_rul], [0, max_rul], "r--", linewidth=2, label="Perfect")
        
        plt.xlabel("Actual RUL", fontsize=11)
        plt.ylabel("Predicted RUL", fontsize=11)
        plt.title(f"{fd} - Predicted vs Actual RUL", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        mae = np.mean(np.abs(df["predicted_rul"] - df["actual_rul"]))
        rmse = np.sqrt(np.mean((df["predicted_rul"] - df["actual_rul"]) ** 2))
        plt.text(0.05, 0.95, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}", 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(args.output_dir / f"predictions_{fd}.png", dpi=150, bbox_inches="tight")
        print(f"Saved {args.output_dir / f'predictions_{fd}.png'}")
        plt.close()


if __name__ == "__main__":
    main()