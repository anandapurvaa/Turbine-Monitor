import pandas as pd
from pathlib import Path

DATA = Path("data/raw")
PROCESSED = Path("data/processed")

# Load raw files
cols = [
    "unit_nr", "time_in_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

train = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv(DATA / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None, names=["RUL"])

# Add RUL to training data (countdown to failure)
def add_rul_train(df):
    df = df.copy()
    max_cycles = df.groupby("unit_nr")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]
    return df

train_rul = add_rul_train(train)

# Add RUL to test data
# The RUL file gives the RUL at the LAST cycle of each test engine
def add_rul_test(df, rul_df):
    df = df.copy()
    # Get the RUL at last cycle for each unit
    rul_at_last = rul_df.copy()
    rul_at_last["unit_nr"] = rul_at_last.index + 1  # unit_nr is 1-indexed
    
    # Get max cycle for each unit in test set
    max_cycles = df.groupby("unit_nr")["time_in_cycles"].transform("max")
    
    # Merge RUL at last cycle
    df = df.merge(rul_at_last, on="unit_nr", how="left")
    
    # Calculate RUL for each row: RUL_at_last + (max_cycle - current_cycle)
    df["RUL"] = df["RUL"] + (max_cycles - df["time_in_cycles"])
    
    return df

test_rul = add_rul_test(test, rul)

# Apply RUL cap
RUL_CAP = 125
train_rul["RUL_capped"] = train_rul["RUL"].clip(upper=RUL_CAP)
test_rul["RUL_capped"] = test_rul["RUL"].clip(upper=RUL_CAP)

# Save
train_rul.to_csv(PROCESSED / "train_FD001_processed.csv", index=False)
test_rul.to_csv(PROCESSED / "test_FD001_processed.csv", index=False)

print("Train shape:", train_rul.shape)
print("Test shape:", test_rul.shape)
print("\nTrain RUL_capped stats:")
print(train_rul["RUL_capped"].describe())
print("\nTest RUL_capped stats:")
print(test_rul["RUL_capped"].describe())
print("\nNaN in test RUL_capped:", test_rul["RUL_capped"].isna().sum())