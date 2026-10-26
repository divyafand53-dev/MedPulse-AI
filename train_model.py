"""
Model Training Script for Medicine Recommendation System.
Trains and compares:
- Random Forest Classifier
- Support Vector Machine (SVC)
- Multinomial Naive Bayes
- Decision Tree Classifier

Saves the best model, label encoder, symptom feature names, and benchmark metrics.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate():
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "Testing.csv"))

    # Separate features and target
    X_train = train_df.drop(columns=["prognosis"])
    y_train = train_df["prognosis"]

    X_test = test_df.drop(columns=["prognosis"])
    y_test = test_df["prognosis"]

    symptom_features = list(X_train.columns)

    # Encode disease labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Candidate Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    metrics = {}
    best_model_name = None
    best_acc = -1.0
    best_model_obj = None

    print("\n--- Training and Evaluating Models ---")
    for name, clf in models.items():
        # Train
        clf.fit(X_train, y_train_enc)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test_enc, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, preds, average="weighted", zero_division=0)
        cv_scores = cross_val_score(clf, X_train, y_train_enc, cv=5)

        metrics[name] = {
            "Accuracy": round(float(acc) * 100, 2),
            "Precision": round(float(prec) * 100, 2),
            "Recall": round(float(rec) * 100, 2),
            "F1-Score": round(float(f1) * 100, 2),
            "CV_Mean_Accuracy": round(float(np.mean(cv_scores)) * 100, 2)
        }

        print(f"[{name}] Test Accuracy: {metrics[name]['Accuracy']}% | 5-Fold CV: {metrics[name]['CV_Mean_Accuracy']}% | F1: {metrics[name]['F1-Score']}%")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = clf

    print(f"\nBest Model: {best_model_name} with {best_acc*100:.2f}% accuracy")

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    features_path = os.path.join(MODELS_DIR, "symptom_features.json")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")

    with open(model_path, "wb") as f:
        pickle.dump(best_model_obj, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    with open(features_path, "w") as f:
        json.dump(symptom_features, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": metrics,
            "classes": list(le.classes_),
            "total_features": len(symptom_features),
            "training_samples": len(X_train),
            "testing_samples": len(X_test)
        }, f, indent=2)

    print("All models and metadata successfully exported to `models/` directory.")

if __name__ == "__main__":
    train_and_evaluate()
