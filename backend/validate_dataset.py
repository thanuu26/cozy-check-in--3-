
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "student_data_filled.csv"

df = pd.read_csv(file_path)

print("Dataset dimensions:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nGenerated low-study records:")
print(df["fake_low_study_row"].value_counts())

print("\nWellbeing statistics:")
print(df["wellbeing"].describe())

required_columns = [
    "sleep", "study", "screen",
    "meals", "activity", "mood", "wellbeing"
]

assert all(col in df.columns for col in required_columns)
assert len(df) == 2800
assert df[required_columns].isnull().sum().sum() == 0

print("\nSUCCESS: Dataset validation passed!")