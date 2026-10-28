
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "cleaned_students.csv"
OUTPUT_DIR = PROJECT_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Pass vs fail distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="result")
plt.title("Student Pass vs Fail Distribution")
plt.xlabel("Result (0 = Fail, 1 = Pass)")
plt.ylabel("Number of Students")
plt.savefig(OUTPUT_DIR / "pass_fail_distribution.png", bbox_inches="tight")
plt.show()

# 2. Final grade distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="G3", bins=15, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (0–20)")
plt.ylabel("Number of Students")
plt.savefig(OUTPUT_DIR / "final_grade_distribution.png", bbox_inches="tight")
plt.show()

# 3. Study time vs final grade
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade")
plt.savefig(OUTPUT_DIR / "studytime_vs_grade.png", bbox_inches="tight")
plt.show()

# 4. Absences vs final grade
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="absences", y="G3", hue="result")
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.savefig(OUTPUT_DIR / "absences_vs_grade.png", bbox_inches="tight")
plt.show()

print("Visualizations saved in:", OUTPUT_DIR)
