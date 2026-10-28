
import pandas as pd
import json
from pathlib import Path

# Locate the project folder
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "student-mat.csv"
JSON_PATH = BASE_DIR / "data" / "students.json"

# Read the CSV file
df = pd.read_csv(CSV_PATH, sep=";")

# Add a unique ID to each student
df.insert(0, "student_id", range(1, len(df) + 1))

# Check the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

# Convert records to JSON
records = df.to_dict(orient="records")

with open(JSON_PATH, "w", encoding="utf-8") as file:
    json.dump(records, file, indent=4, ensure_ascii=False)

print("\nJSON created successfully!")
print("Total student records:", len(records))
print("Saved to:", JSON_PATH)
