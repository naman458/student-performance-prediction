
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load credentials from .env
load_dotenv(PROJECT_DIR / ".env")
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env")

# Connect to MongoDB Atlas
client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

try:
    client.admin.command("ping")
    print("MongoDB connection successful!")

    db = client["student_performance_db"]
    students = db["students"]

    print("Database:", db.name)
    print("Collection:", students.name)

except Exception as error:
    print("Connection failed:", error)

finally:
    client.close()
