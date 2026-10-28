
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "cleaned_students.csv"
OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Target: 0 = Fail, 1 = Pass
y = df["result"]

# Exclude identifiers and grade columns to prevent target leakage
X = df.drop(columns=["student_id", "G1", "G2", "G3", "result"])

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Preprocessing is fitted only on training data by the pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Held-out test set: 20% of the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# Cross-validation evaluates both classes more fairly
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "fail_recall": make_scorer(
        recall_score,
        pos_label=0,
        zero_division=0
    )
}

results = []
sns.set_theme(style="whitegrid")

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Cross-validation uses only the training portion
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # Fit final pipeline and evaluate on untouched test data
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Preserve existing metrics: these refer to the Pass class
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Class-specific metrics
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Fail Precision": report["Fail"]["precision"],
        "Fail Recall": report["Fail"]["recall"],
        "Fail F1": report["Fail"]["f1-score"],
        "Macro F1": report["macro avg"]["f1-score"],
        "CV Accuracy Mean": cv_results["test_accuracy"].mean(),
        "CV Accuracy Std": cv_results["test_accuracy"].std(),
        "CV Macro F1 Mean": cv_results["test_macro_f1"].mean(),
        "CV Macro F1 Std": cv_results["test_macro_f1"].std(),
        "CV Fail Recall Mean": cv_results["test_fail_recall"].mean()
    })

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)
    print(f"Test Accuracy:       {accuracy:.3f}")
    print(f"Pass Precision:      {precision:.3f}")
    print(f"Pass Recall:         {recall:.3f}")
    print(f"Pass F1:             {f1:.3f}")
    print(f"Fail Recall:         {report['Fail']['recall']:.3f}")
    print(f"Macro F1:            {report['macro avg']['f1-score']:.3f}")
    print(
        "5-fold CV Accuracy:  "
        f"{cv_results['test_accuracy'].mean():.3f} "
        f"+/- {cv_results['test_accuracy'].std():.3f}"
    )
    print(
        "5-fold CV Macro F1:  "
        f"{cv_results['test_macro_f1'].mean():.3f} "
        f"+/- {cv_results['test_macro_f1'].std():.3f}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Result")
    plt.ylabel("Actual Result")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # Save the trained pipeline using the existing filename
    joblib.dump(pipeline, OUTPUT_DIR / f"{filename}.joblib")

# Save detailed metrics
results_df = pd.DataFrame(results)

# Compare models primarily on balanced performance across both classes
results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)

print("\n\nMODEL COMPARISON")
print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.3f}"
))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# Keep the comparison chart readable
chart_metrics = [
    "Accuracy",
    "F1 Score",
    "Fail Recall",
    "Macro F1"
]

plot_df = results_df[["Model"] + chart_metrics].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model"
)
plt.ylim(0, 1)
plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nResults saved to:", OUTPUT_DIR)
print(
    "Updated models, comparison CSV/chart, "
    "and confusion matrices saved successfully."
)
