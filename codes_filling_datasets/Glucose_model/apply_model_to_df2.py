import pandas as pd
from joblib import load
from pathlib import Path
import xgboost as xgb

script_dir = Path(__file__).parent

# load model
XGB = load(script_dir / "xgb_glucose_model.joblib")

# load new data
df2 = pd.read_csv(script_dir.parent.parent / "cleaned_datasets" / "dataset2.csv")

df2_copy = df2[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Outcome"
    ]
]

df2["Glucose"] = XGB.predict(df2_copy)

df2.to_csv(script_dir / "dataset2_filled.csv", index = False)