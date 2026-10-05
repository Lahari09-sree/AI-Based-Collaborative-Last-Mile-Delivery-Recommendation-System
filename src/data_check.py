import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "delivery_logistics.csv"
OUTPUT_FILE = BASE_DIR / "data" / "delivery_logistics_clean.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 50)
print("ORIGINAL DATA")
print("=" * 50)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nOriginal data types:")
print(df.dtypes)


# ============================================================
# FIX DELIVERY TIME COLUMNS
# ============================================================

def convert_time_to_hours(series):
    """
    Convert delivery time values into numeric hours.

    Handles:
    - normal numeric values such as 8, 10, 12
    - strings such as '8'
    - corrupted datetime strings such as
      '1970-01-01 00:00:00.000000008'
    """

    # First try numeric conversion
    numeric = pd.to_numeric(series, errors="coerce")

    # Identify values that could not be converted normally
    mask = numeric.isna()

    if mask.any():

        # Convert problematic values to datetime
        dates = pd.to_datetime(
            series[mask],
            errors="coerce"
        )

        # Extract nanoseconds from Unix epoch
        extracted = dates.astype("int64") % 1_000_000_000

        # Convert nanoseconds back to the intended numeric value
        numeric.loc[mask] = extracted.astype(float)

    return numeric


df["delivery_time_hours"] = convert_time_to_hours(
    df["delivery_time_hours"]
)

df["expected_time_hours"] = convert_time_to_hours(
    df["expected_time_hours"]
)


# ============================================================
# CHECK TIME VALUES
# ============================================================

print("\nSample delivery_time_hours values:")
print(df["delivery_time_hours"].head(20).tolist())

print("\nSample expected_time_hours values:")
print(df["expected_time_hours"].head(20).tolist())


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "delivery_id",
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

required_columns = [
    "delivery_time_hours",
    "expected_time_hours",
    "distance_km",
    "package_weight_kg",
    "delivery_cost"
]

before = len(df)

df = df.dropna(
    subset=required_columns
)

after = len(df)

print("\nInvalid rows removed:", before - after)


# ============================================================
# REMOVE IMPOSSIBLE VALUES
# ============================================================

df = df[
    (df["delivery_time_hours"] > 0) &
    (df["expected_time_hours"] > 0) &
    (df["distance_km"] > 0) &
    (df["package_weight_kg"] > 0) &
    (df["delivery_cost"] > 0)
].copy()


# ============================================================
# DELAY BINARY
# ============================================================

df["delayed_binary"] = (
    df["delayed"]
    .astype(str)
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })
)

# If delayed column contains something unexpected,
# derive delay from actual vs expected time.

df["delayed_binary"] = df["delayed_binary"].fillna(
    (df["delivery_time_hours"] > df["expected_time_hours"]).astype(int)
)


# ============================================================
# TIME DIFFERENCE
# ============================================================

df["time_difference"] = (
    df["delivery_time_hours"]
    - df["expected_time_hours"]
)


# ============================================================
# COST PER KM
# ============================================================

df["cost_per_km"] = (
    df["delivery_cost"]
    / df["distance_km"]
)


# ============================================================
# DELIVERY EFFICIENCY
# ============================================================

# Higher value = better efficiency
#
# Distance / Time gives speed.
# We additionally divide by cost so that lower cost
# gives better efficiency.

df["delivery_efficiency"] = (
    df["distance_km"]
    / (
        df["delivery_time_hours"]
        * df["delivery_cost"]
    )
)


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# FINAL MISSING VALUE HANDLING
# ============================================================

df = df.dropna().copy()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 50)
print("AFTER CLEANING")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("CLEANING COMPLETED")
print("=" * 50)

print("Final shape:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nClean dataset saved to:")
print(OUTPUT_FILE)