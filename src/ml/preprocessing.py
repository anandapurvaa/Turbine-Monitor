import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

cols = [
    "unit_nr", "time_in_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

# Process all 4 datasets
datasets = ["FD001", "FD002", "FD003", "FD004"]

for fd in datasets:
    print(f"\nProcessing {fd}...")
    
    train = pd.read_csv(DATA / f"train_{fd}.txt", sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(DATA / f"test_{fd}.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(DATA / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"])
    
    # Add RUL to training data
    def add_rul_train(df):
        df = df.copy()
        max_cycles = df.groupby("unit_nr")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycles - df["time_in_cycles"]
        return df
    
    train_rul = add_rul_train(train)
    
    # Add RUL to test data
    def add_rul_test(df, rul_df):
        df = df.copy()
        rul_at_last = rul_df.copy()
        rul_at_last["unit_nr"] = rul_at_last.index + 1
        max_cycles = df.groupby("unit_nr")["time_in_cycles"].transform("max")
        df = df.merge(rul_at_last, on="unit_nr", how="left")
        df["RUL"] = df["RUL"] + (max_cycles - df["time_in_cycles"])
        return df
    
    test_rul = add_rul_test(test, rul)
    
    # Apply RUL cap
    RUL_CAP = 125
    train_rul["RUL_capped"] = train_rul["RUL"].clip(upper=RUL_CAP)
    test_rul["RUL_capped"] = test_rul["RUL"].clip(upper=RUL_CAP)
    
    # Split train into train/val (80/20 of units, not rows)
    unique_units = train_rul["unit_nr"].unique()
    train_units, val_units = train_test_split(unique_units, test_size=0.2, random_state=42)
    
    train_split = train_rul[train_rul["unit_nr"].isin(train_units)]
    val_split = train_rul[train_rul["unit_nr"].isin(val_units)]
    
    # Save
    train_split.to_csv(PROCESSED / f"train_{fd}_processed.csv", index=False)
    val_split.to_csv(PROCESSED / f"val_{fd}_processed.csv", index=False)
    test_rul.to_csv(PROCESSED / f"test_{fd}_processed.csv", index=False)
    
    print(f"Train: {train_split.shape}, Val: {val_split.shape}, Test: {test_rul.shape}")

print("\n✅ All datasets processed!")