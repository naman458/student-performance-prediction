
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Find project folder and load connection string
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
JSON_PATH = PROJECT_DIR / "data" / "students.json"

# Read the JSON dataset
with open(JSON_PATH, "r", encoding="utf-8") as file:
    records = json.load(file)

print("Records in JSON:", len(records))

# Connect to MongoDB
client = MongoClient(MONGODB_URI)

try:
    client.admin.command("ping")

    db = client["student_performance_db"]
    students = db["students"]

    existing_count = students.count_documents({})

    if existing_count == 0:
        if records:
            result = students.insert_many(records)
            print("Inserted documents:", len(result.inserted_ids))
        else:
            print("JSON file is empty.")
    else:
        print("Collection already contains", existing_count, "documents.")
        print("Skipping insertion to avoid duplicate records.")

    print("Total documents in MongoDB:", students.count_documents({}))

finally:
    client.close()
