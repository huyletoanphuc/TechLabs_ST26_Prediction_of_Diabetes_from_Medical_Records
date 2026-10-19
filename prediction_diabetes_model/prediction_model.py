import numpy as np
import pandas as pd
import matplotlib as plt
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

script_dir = Path(__file__).parent

df = pd.read_csv(script_dir.parent / "combining_into_1_csv" / "combined_dataset.csv")

print("\nOutcome counts:")
print(df["Outcome"].value_counts())

x = df[
    [
    "Age",
    "Gender",
    "BMI",
    "Glucose",
    "Insulin"
    ]
]

y = df["Outcome"].astype(int)

print("X shape :" , x.shape)
print("Y shape :" , y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state=42, stratify = y)

print("Training samples:" , len(x_train))
print("Testing samples:", len(x_test))

model = Pipeline([
    ("imputer" , SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(x_train, y_train)
y_probability = model.predict_proba(x_test)[:, 1]

CUSTOM_THRESHOLD = 0.50
y_pred = (y_probability >= CUSTOM_THRESHOLD).astype(int)

y_pred = model.predict(x_test)

print("\nFirst 10 Class Predictions (at threshold = 0.20):")
print(y_pred[:10])

print("\nFirst 10 Predicted Probabilities:")
print(y_probability[:10])

#Evaluation 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(
    y_test, y_pred, zero_division=0
))

print("Recall / Sensitivity:", recall_score(
    y_test, y_pred, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, y_pred, zero_division=0
))

print("ROC-AUC:", roc_auc_score(
    y_test, y_probability
))

print("PR-AUC:", average_precision_score(
    y_test, y_probability
))

print("Log Loss:", log_loss(
    y_test, y_probability
))

print("Brier Score:", brier_score_loss(
    y_test, y_probability
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

results = x_test.copy()

results["Actual_Outcome"] = y_test.values
results["Diabetes_Probability"] = y_probability
results["Predicted_Outcome"] = y_pred

print(
    results.sort_values(
        "Diabetes_Probability",
        ascending=False
    ).head(20)
)

# plot the ROC curve 
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc = roc_auc_score(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Diabetes Prediction")
plt.legend()
plt.grid()

plt.show()

joblib.dump(model, script_dir / "diabetes_model.joblib")

print("model saved successfully!")