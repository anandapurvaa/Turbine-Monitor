import pandas as pd

for fd in ["FD001", "FD002", "FD003", "FD004"]:
    path = f"data/processed/train_{fd}_processed.csv"
    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(fd)
    print("=" * 60)

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head(3).to_string())

    possible_settings = [
        col for col in df.columns
        if "setting" in col.lower() or "op" in col.lower()
    ]

    print("\nPossible operating-condition columns:")
    print(possible_settings if possible_settings else "NONE FOUND")

    if possible_settings:
        print("\nUnique operating-condition rows:")
        print(df[possible_settings].drop_duplicates().shape[0])

        print("\nSample conditions:")
        print(df[possible_settings].drop_duplicates().head(10).to_string(index=False))

    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    print(f"\nSensor columns found: {len(sensor_cols)}")
    print(f"Missing values: {df.isna().sum().sum()}")