# ==========================================
# 01_data_preparation.py
# ==========================================

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================
INPUT_PATH = "data/tourism_data.csv"
OUTPUT_DIR = "data"
TARGET_COL = "ProdTaken"
TEST_SIZE = 0.25
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOGGER
# ==========================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(OUTPUT_DIR, "complete_output.txt"))

print("="*70)
print("DATA PREPARATION STARTED")
print("="*70)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING")
print("="*70)

df = pd.read_csv(INPUT_PATH)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn names:")
print(df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {INPUT_PATH}")

# ==========================================
# STEP 2: TARGET IMBALANCE CHECK
# ==========================================
print("\n" + "="*70)
print("STEP 2: TARGET DISTRIBUTION (IMBALANCE CHECK)")
print("="*70)

print(df[TARGET_COL].value_counts(dropna=False))
print("\nPercentage:")
print(df[TARGET_COL].value_counts(normalize=True, dropna=False))

# ==========================================
# STEP 3: MISSING VALUE CHECK
# ==========================================
# ==========================================
# STEP 3: MISSING VALUES HANDLING (median/mode)
# ==========================================
print("\n" + "="*70)
print("STEP 3: MISSING VALUES HANDLING")
print("="*70)

# 3A) Report missing values BEFORE handling (for documentation)
missing_before = df.isnull().sum().sort_values(ascending=False)
print("\nMissing values BEFORE handling:")
print(missing_before[missing_before > 0])

missing_before.to_csv(os.path.join(OUTPUT_DIR, "missing_values_report_before.csv"))

# 3B) Fill missing values
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Numeric -> median
for col in num_cols:
    if df[col].isna().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled numeric '{col}' with median = {median_val}")

# Categorical -> mode (most frequent)
for col in cat_cols:
    if df[col].isna().any():
        mode_series = df[col].mode(dropna=True)
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "unknown"
        df[col] = df[col].fillna(mode_val)
        print(f"Filled categorical '{col}' with mode = {mode_val}")

# 3C) Confirm after
missing_after = df.isnull().sum().sort_values(ascending=False)
print("\nMissing values AFTER handling:")
print(missing_after[missing_after > 0])

missing_after.to_csv(os.path.join(OUTPUT_DIR, "missing_values_report_after.csv"))


# ==========================================
# STEP 4: DUPLICATE CHECK
# ==========================================
print("\n" + "="*70)
print("STEP 4: DUPLICATE CHECK")
print("="*70)

dup_count = df.duplicated().sum()
print(f"Total duplicate rows: {dup_count}")

if dup_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")
    print("New shape:", df.shape)

# ==========================================
# STEP 5: FEATURE ENGINEERING - CLEANING
# ==========================================
print("\n" + "="*70)
print("STEP 5: FEATURE ENGINEERING - CLEANING")
print("="*70)

# IMPORTANT CHANGE:
# Do NOT do .astype(str) because it turns NaN into "nan" strings.
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    df[col] = (
        df[col]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

# Targeted mapping fixes (keep as you wrote, but safe with NaNs)
if "Gender" in df.columns:
    print("\nFixing Gender inconsistencies...")
    df["Gender"] = df["Gender"].where(df["Gender"].isna(), df["Gender"].str.replace(" ", "", regex=False))
    df["Gender"] = df["Gender"].replace({
        "f": "female",
        "m": "male",
        "fe male": "female"
    })
    print("Unique Gender values:", df["Gender"].dropna().unique())

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"unmarried": "single"})

# ==========================================
# STEP 6: DurationOfPitch OUTLIER (keep informational only)
# ==========================================
print("\n" + "="*70)
print("STEP 6: DurationOfPitch OUTLIER ANALYSIS")
print("="*70)

if "DurationOfPitch" in df.columns:
    print("Min DurationOfPitch:", df["DurationOfPitch"].min())
    print("Max DurationOfPitch:", df["DurationOfPitch"].max())

    extreme_values = df[df["DurationOfPitch"] > 60]
    print(f"Rows with DurationOfPitch > 60: {len(extreme_values)}")

    if len(extreme_values) > 0:
        print("\nSample of extreme values:")
        cols_to_show = ["DurationOfPitch"]
        if TARGET_COL in df.columns:
            cols_to_show.append(TARGET_COL)
        print(extreme_values[cols_to_show].head())

    print("\nDecision: Keep values (no removal).")

# ==========================================
# STEP 7: TRAIN/TEST SPLIT (75/25)
# ==========================================
print("\n" + "="*70)
print("STEP 7: TRAIN/TEST SPLIT")
print("="*70)

# Drop rows with missing target (required for stratify + training)
before = len(df)
df = df.dropna(subset=[TARGET_COL])
after = len(df)
print(f"Dropped rows with missing target: {before - after}")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COL].astype(int) if df[TARGET_COL].dtype != "int64" else df[TARGET_COL]
)

print(f"\nTrain set: {train_df.shape[0]} rows")
print(f"Test set: {test_df.shape[0]} rows")

print("\nTrain Target Distribution:")
print(train_df[TARGET_COL].value_counts(normalize=True))

print("\nTest Target Distribution:")
print(test_df[TARGET_COL].value_counts(normalize=True))

# ==========================================
# STEP 8: SAVE
# ==========================================
print("\n" + "="*70)
print("STEP 8: SAVING CLEANED DATA")
print("="*70)

train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print("\nFiles saved:")
print("data/train_data.csv")
print("data/test_data.csv")

print("\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)

# Close logger
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
