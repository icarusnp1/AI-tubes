from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


FEATURES_DEFAULT = [
    "accuracy",
    "num_steps",
    "num_incorrect_steps",
    "avg_step_time",
    "max_step_time",
    "total_time",
    "error_streak_max",
    "repeat_error_rate",
    "unique_skills",
]


def split_by_student(df: pd.DataFrame, student_col: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    students = df[student_col].astype(str).unique()
    train_students, temp_students = train_test_split(students, test_size=(test_size + val_size), random_state=seed)
    # split temp into val and test
    rel_test = test_size / (test_size + val_size)
    val_students, test_students = train_test_split(temp_students, test_size=rel_test, random_state=seed)

    train_df = df[df[student_col].astype(str).isin(train_students)].copy()
    val_df = df[df[student_col].astype(str).isin(val_students)].copy()
    test_df = df[df[student_col].astype(str).isin(test_students)].copy()
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sessions.csv", help="Path to session-level CSV.")
    ap.add_argument("--student_col", default="student_id", help="Student id column name (after preprocess).")
    ap.add_argument("--label_col", default="understanding_class", help="Label column.")
    ap.add_argument("--model_out", default="models/model.joblib", help="Output model path.")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--min_samples_leaf", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)

    # Ensure student col exists
    if args.student_col not in df.columns:
        # try common alternatives (since preprocess uses original col name)
        candidates = [c for c in df.columns if "student" in c and "id" in c]
        raise ValueError(f"student_col '{args.student_col}' not found. Candidates: {candidates}")

    # Choose features present
    features = [f for f in FEATURES_DEFAULT if f in df.columns]
    if not features:
        raise ValueError(f"No expected features found. Available columns: {list(df.columns)}")

    # Drop NA rows for features/label
    df = df.dropna(subset=features + [args.label_col])

    train_df, val_df, test_df = split_by_student(df, args.student_col, seed=args.seed)

    X_train = train_df[features].to_numpy()
    y_train = train_df[args.label_col].astype(str).to_numpy()

    X_val = val_df[features].to_numpy()
    y_val = val_df[args.label_col].astype(str).to_numpy()

    X_test = test_df[features].to_numpy()
    y_test = test_df[args.label_col].astype(str).to_numpy()

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "features": features,
        "params": {
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "seed": args.seed,
        }
    }

    # Reports
    report_text = classification_report(y_test, test_pred, digits=4)
    cm = confusion_matrix(y_test, test_pred, labels=["LOW", "MED", "HIGH"])

    # Save outputs
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, model_out)

    Path("models").mkdir(exist_ok=True)
    with open("models/feature_list.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=["LOW", "MED", "HIGH"], columns=["LOW", "MED", "HIGH"]).to_csv("reports/confusion_matrix.csv")

    rules = export_text(clf, feature_names=features)
    with open("reports/tree_rules.txt", "w", encoding="utf-8") as f:
        f.write(rules + "\n\n")
        f.write("TEST REPORT:\n")
        f.write(report_text)

    print("Saved model:", model_out)
    print("Saved reports to reports/")
    print("Validation:", metrics["val_accuracy"], metrics["val_macro_f1"])
    print("Test:", metrics["test_accuracy"], metrics["test_macro_f1"])
    print("\nTest classification report:\n", report_text)


if __name__ == "__main__":
    main()