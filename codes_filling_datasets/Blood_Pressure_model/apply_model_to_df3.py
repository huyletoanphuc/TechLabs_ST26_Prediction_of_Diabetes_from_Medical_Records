import pandas as pd
from joblib import load
from pathlib import Path
import xgboost as xgb

script_dir = Path(__file__).parent

XGB = load(script_dir / "xgb_blood_pressure_model.joblib")

df3 = pd.read_csv(script_dir.parent.parent /"cleaned_datasets" /"dataset3.csv")

dataset3 = df3[
    [
        "Age",
        "BMI",
        "Glucose",
        "Outcome"
    ]
]

df3["BloodPressure"] = XGB.predict(dataset3)

print(df3.head())
print(df3.columns.tolist())

df3.to_csv(script_dir / "dataset3_filled.csv", index = False)