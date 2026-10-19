import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from pathlib import Path

from joblib import dump

script_dir = Path(__file__).parent

df = pd.read_csv(script_dir.parent.parent / "cleaned_datasets" / "dataset1.csv")

df = df[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Outcome"
    ]
]

X = df[["Age" , "Gender" , "BMI" , "BloodPressure", "Outcome"]]
Y = df['Glucose']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Model training:
model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0
)
model.fit(X_train,y_train)

dump(model, script_dir / "xgb_glucose_model.joblib")

predictions = model.predict(X_test)

# Error evalutation:
errors = predictions - y_test

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
bias = np.mean(errors)
corr, _ = pearsonr(predictions, y_test)

std_dev = np.std(errors)

loa_upper = bias + 1.96 * std_dev
loa_lower = bias - 1.96 * std_dev

print("\nModel Evaluation")
print("=" * 50)
print(f"RMSE:                {rmse:.3f}")
print(f"MAE:                 {mae:.3f}")
print(f"Bias:                {bias:.3f}") # Shows whether the model systematically over- or under-predicts( positive -> overestimates , negative -> underestimate)
print(f"Correlation (r):     {corr:.4f}") # values close to 1 -> strong pos agreement, values neer 0 -> little relationship, negative values -> an inverse relationship
print(f"Upper LOA:           {loa_upper:.3f}")
print(f"Lower LOA:           {loa_lower:.3f}")

# Scatter Plot:
plt.figure(figsize = (6,5))

plt.scatter(y_test, predictions, alpha = 0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label="Perfect Prediction"
)

plt.xlabel("Actual Blood Pressure")
plt.ylabel("Predicted Blood Pressure")
plt.title("Predicted vs Actual Blood Pressure")
plt.legend()
plt.grid(True)

plt.show()

# ----------------------------
# Bland-Altman Plot
# ----------------------------

mean_values = (predictions + y_test) / 2
differences = predictions - y_test

plt.figure(figsize=(6, 5))

plt.scatter(mean_values, differences, alpha=0.6)

plt.axhline(
    bias,
    color="green",
    linestyle="-",
    label=f"Bias = {bias:.2f}"
)

plt.axhline(
    loa_upper,
    color="gray",
    linestyle="--",
    label="Upper LOA"
)

plt.axhline(
    loa_lower,
    color="gray",
    linestyle="--",
    label="Lower LOA"
)

plt.xlabel("Mean of Prediction and Actual")
plt.ylabel("Prediction - Actual")
plt.title("Bland-Altman Plot")
plt.legend()
plt.grid(True)

plt.show()