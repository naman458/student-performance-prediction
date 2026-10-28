
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

# 1. Set up paths and MongoDB connection
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["student_performance_db"]
students = db["students"]

try:
    client.admin.command("ping")

    # 2. Retrieve records from MongoDB
    records = list(students.find({}, {"_id": 0}))
    df = pd.DataFrame(records)

    print("Original dataset shape:", df.shape)

    # 3. Inspect missing values and duplicates
    print("\nMissing values before cleaning:")
    print(df.isnull().sum().sum())

    print("Duplicate student IDs:",
          df["student_id"].duplicated().sum())

    print("Duplicate rows:",
          df.drop(columns=["student_id"]).duplicated().sum())

    # 4. Remove duplicate student IDs
    df = df.drop_duplicates(subset=["student_id"], keep="first")

    # 5. Validate final grades
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
    df.loc[~df["G3"].between(0, 20), "G3"] = float("nan")

    # Remove records where the target cannot be determined
    df = df.dropna(subset=["G3"])

    # 6. Handle missing values in remaining columns
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_values = df[col].mode()
            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])

    # 7. Create the prediction target
    df["result"] = (df["G3"] >= 10).astype(int)

    # 8. Remove grade columns from model input data
    # Keep G3 only for defining the target, not as a feature.
    model_df = df.drop(columns=["G1", "G2", "G3", "student_id"])

    # 9. Save cleaned data and ML-ready data
    cleaned_path = PROJECT_DIR / "data" / "cleaned_students.csv"
    model_path = PROJECT_DIR / "data" / "processed_students.csv"

    df.to_csv(cleaned_path, index=False)
    model_df.to_csv(model_path, index=False)

    # 10. Report results
    print("\nFinal cleaned dataset shape:", df.shape)
    print("Missing values after cleaning:", df.isnull().sum().sum())

    print("\nPass/fail distribution (0 = fail, 1 = pass):")
    print(df["result"].value_counts().sort_index())

    print("\nSaved cleaned data to:", cleaned_path)
    print("Saved ML-ready data to:", model_path)

finally:
    client.close()
