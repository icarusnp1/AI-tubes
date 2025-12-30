import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

DATA_FILE = "data/processed/sessions.csv"
MODEL_OUT = "models/struggle_tree.joblib"

df = pd.read_csv(DATA_FILE)

# ===== Define STRUGGLE label =====
df["struggle_label"] = (
    (df["accuracy"] < 0.6) |
    (df["error_streak_max"] >= 3) |
    (df["repeat_error_rate"] >= 0.3)
).map({True: "STRUGGLE", False: "OK"})

FEATURES = [
    "num_steps",
    "avg_step_time",
    "max_step_time",
    "total_time",
    "error_streak_max",
    "repeat_error_rate",
]

X = df[FEATURES]
y = df["struggle_label"]

# ===== Split by student =====
students = df["student_id"].unique()
train_students, test_students = train_test_split(
    students, test_size=0.2, random_state=42
)

train_idx = df["student_id"].isin(train_students)
test_idx = df["student_id"].isin(test_students)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ===== Train tree =====
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=200,
    class_weight="balanced",
    random_state=42,
)

clf.fit(X_train, y_train)

# ===== Evaluate =====
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=["OK", "STRUGGLE"]))

# ===== Save =====
os.makedirs("models", exist_ok=True)
dump(clf, MODEL_OUT)

rules = export_text(clf, feature_names=FEATURES)
with open("models/struggle_tree_rules.txt", "w") as f:
    f.write(rules)

print("Model saved to:", MODEL_OUT)
