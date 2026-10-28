
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# Load MongoDB connection string
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["student_performance_db"]
students = db["students"]

# Temporary test student; does not affect original dataset
test_student = {
    "student_id": 9999,
    "school": "GP",
    "sex": "M",
    "age": 18,
    "studytime": 2,
    "absences": 0,
    "G1": 10,
    "G2": 11,
    "G3": 12
}

try:
    client.admin.command("ping")

    # CREATE: Insert a test document
    students.delete_one({"student_id": 9999})
    result = students.insert_one(test_student)
    print("\n1. CREATE")
    print("Inserted document ID:", result.inserted_id)

    # READ: Display the first five student records
    print("\n2. READ")
    for student in students.find({}, {"_id": 0}).limit(5):
        print(student)

    # Read the test student
    print("\nTest student before update:")
    print(students.find_one(
        {"student_id": 9999},
        {"_id": 0}
    ))

    # UPDATE: Change the test student's absences
    result = students.update_one(
        {"student_id": 9999},
        {"$set": {"absences": 3}}
    )
    print("\n3. UPDATE")
    print("Documents modified:", result.modified_count)
    print("Updated student:", students.find_one(
        {"student_id": 9999},
        {"_id": 0}
    ))

    # DELETE: Remove only the test student
    result = students.delete_one({"student_id": 9999})
    print("\n4. DELETE")
    print("Documents deleted:", result.deleted_count)

    # Verify the original dataset remains intact
    print("\nFinal dataset document count:",
        students.count_documents({}))

finally:
    client.close()
