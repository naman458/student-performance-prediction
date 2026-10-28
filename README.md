# Student Performance Prediction System

## Overview

The Student Performance Prediction System is a Python-based machine learning application that analyzes student-related attributes and predicts whether a student is likely to pass or fail. It combines data preprocessing, machine learning, visualizations, and MongoDB database operations in an interactive Streamlit dashboard.

## Features

- **Data preprocessing:** Handles missing values and prepares student data for machine learning.
- **Machine learning:** Implements Logistic Regression and Random Forest classifiers.
- **Model evaluation:** Compares accuracy, precision, recall, F1-score, macro F1, and cross-validation results.
- **Interactive dashboard:** Explores the dataset and displays visualizations and model comparisons.
- **Student prediction:** Predicts pass/fail outcomes from entered student attributes.
- **MongoDB integration:** Stores and retrieves student records.
- **CRUD operations:** Supports creating, reading, updating, and deleting student records.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib, Seaborn, and Plotly
- MongoDB Atlas and PyMongo
- Joblib

## Project Structure

```text
student-performance-project/
├── app.py
├── convert_data.py
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   ├── student-mat.csv
│   ├── students.json
│   ├── cleaned_students.csv
│   └── processed_students.csv
├── src/
│   ├── database.py
│   ├── insert_data.py
│   ├── crud_operations.py
│   ├── preprocessing.py
│   ├── visualizations.py
│   └── train_models.py
├── results/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── confusion matrices
└── visualizations/
```

## Setup

### 1. Install Python

Install a compatible Python version and ensure `python` and `pip` work in your terminal.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure MongoDB

Create a `.env` file in the project root with your MongoDB Atlas connection string:

```env
MONGODB_URI=your_mongodb_connection_string
```

Replace the placeholder with your own connection string. Never commit your actual `.env` file or database credentials.

The MongoDB database used by this project is `student_performance_db`, with the `students` collection.

### 4. Prepare the data

If the original dataset needs conversion to JSON, run:

```bash
python convert_data.py
```

Follow with the preprocessing and data insertion scripts as required by your existing setup.

### 5. Train the models

```bash
python src/train_models.py
```

The trained model files and evaluation results are saved in the `results/` directory.

### 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Evaluation

The system compares Logistic Regression and Random Forest using a stratified train/test split and five-fold cross-validation on the training portion.

The current held-out test results are:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 65.8% | 68.4% |
| Pass F1-score | 0.761 | 0.797 |
| Fail recall | 34.6% | 19.2% |
| Macro F1-score | 0.581 | 0.541 |

Random Forest has higher accuracy, while Logistic Regression performs better on macro F1 and identifying failing students in this test split. Both models have limitations, especially in detecting students who fail.

## Dataset and Limitations

The project uses student performance data and defines the target as Pass or Fail based on the project's preprocessing rule. Final-grade columns are excluded from the predictors to reduce target leakage.

Predictions are experimental estimates, not guarantees of academic outcomes. Model performance depends on the dataset and should be validated before real-world use.

## Security

- Keep `.env` private.
- Do not include credentials in source code or submission archives.
- Configure MongoDB Atlas access permissions appropriately.
