from pathlib import Path
import pandas as pd
import csv

file_path = Path(__file__).parent / "student_lifestyle_dataset.csv"

df = pd.read_csv(
    file_path,
    sep=",",
    quoting=csv.QUOTE_NONE,
    quotechar='"'
)

# Remove surrounding quotation marks from column names
df.columns = df.columns.str.strip('"')

# Remove surrounding quotation marks from text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip('"')

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())