"""
train_model.py
==============
Multi-Disease Machine Learning Training and Evaluation Pipeline.
Trains a Multi-Class Random Forest Classifier on the 132-symptom benchmark dataset
to predict across 41 distinct human diseases.
Reports genuine, computed evaluation metrics and saves artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "symptoms_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "symptom_disease_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "symptom_features.json")
REPORT_PATH = os.path.join(MODELS_DIR, "symptom_evaluation_report.json")


def main():
    print("[*] Starting Multi-Disease Symptom Model Training Pipeline...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded symptoms dataset: {df.shape[0]} patient cases, {df.shape[1]} columns.")

    # Target is 'prognosis'
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"].str.strip()

    symptom_features = X.columns.tolist()
    distinct_diseases = sorted(y.unique().tolist())
    print(f"[+] Total symptoms: {len(symptom_features)}")
    print(f"[+] Total target diseases: {len(distinct_diseases)}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train split: {len(X_train)} samples, Test split: {len(X_test)} samples.")

    # Train Multi-Class Random Forest Classifier
    print("[+] Training Multi-Class Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print("\n==================================================")
    print(" Multi-Disease Model Performance Metrics")
    print("==================================================")
    print(f" Test Accuracy : {acc * 100:.2f}%")
    print(f" Precision     : {prec * 100:.2f}%")
    print(f" Recall        : {rec * 100:.2f}%")
    print(f" F1-Score      : {f1 * 100:.2f}%")
    print(f" Total Diseases: {len(distinct_diseases)}")
    print("==================================================")

    # Save model artifact
    joblib.dump(clf, MODEL_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")

    # Save symptom feature list
    with open(FEATURES_PATH, "w") as f:
        json.dump(symptom_features, f, indent=2)
    print(f"[+] Feature dictionary saved to: {FEATURES_PATH}")

    # Save evaluation report JSON
    report_data = {
        "model_name": "Multi-Class Random Forest Disease Classifier",
        "dataset": "Disease-Symptom Knowledge Base (4,920 records)",
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_symptoms": len(symptom_features),
        "num_diseases": len(distinct_diseases),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "diseases": distinct_diseases
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Evaluation report saved to: {REPORT_PATH}")
    print("\n[*] Training completed successfully!")


if __name__ == "__main__":
    main()
