"""Evaluate the SAVED model on the notebook's held-out test split (no retraining).

Rebuilds the exact 80:20 split from Final.ipynb (dropna, LabelEncoder,
train_test_split(test_size=0.2, random_state=42)) and scores the shipped
sentence_classifier_pipeline.pkl on the test portion.
"""
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from srs_classifier import DATASET_PATH, get_classifier


def evaluate():
    clf = get_classifier()
    data = pd.read_csv(DATASET_PATH).dropna()
    encoded = clf.label_encoder.transform(data["Label"])
    _, X_test, _, y_test = train_test_split(
        data["Sentence"], encoded, test_size=0.2, random_state=42)
    y_pred = clf.pipeline.predict(X_test)
    labels = list(range(len(clf.label_encoder.classes_)))
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0)
    return {
        "per_class": [
            {"label": name, "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(clf.label_encoder.classes_)],
        "macro_f1": float(f.mean()),
        "n_rows": len(data),
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, labels=labels,
            target_names=clf.label_encoder.classes_, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": list(clf.label_encoder.classes_),
    }


if __name__ == "__main__":
    r = evaluate()
    print(f"Dataset rows after dropna: {r['n_rows']}  |  test rows: {r['n_test']}")
    print(f"Model Accuracy: {r['accuracy'] * 100:.2f}%")
    print("Classification Report:\n", r["report"])
    print("Confusion matrix (rows = true, cols = predicted):")
    print("Labels:", r["labels"])
    print(r["confusion_matrix"])
