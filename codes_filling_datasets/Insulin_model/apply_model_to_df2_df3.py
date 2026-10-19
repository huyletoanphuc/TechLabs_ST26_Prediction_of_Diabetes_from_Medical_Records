import pandas as pd
from joblib import load
from pathlib import Path
import xgboost as xgb

script_dir = Path(__file__).parent

# load model
XGB = load(script_dir / "xgb_insulin_model.joblib")

# load new data
df2 = pd.read_csv(script_dir.parent / "Glucose_model" / "dataset2_filled.csv")

df2_copy = df2[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Outcome"
    ]
]

df2["Insulin"] = XGB.predict(df2_copy)

df2.to_csv(script_dir.parent.parent / "cleaned_datasets" / "dataset2_filled.csv", index = False)




# load new data
df3 = pd.read_csv(script_dir.parent / "Blood_Pressure_model" / "dataset3_filled.csv")

df3_copy = df3[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Outcome"
    ]
]

df3["Insulin"] = XGB.predict(df3_copy)

df3.to_csv(script_dir.parent.parent / "cleaned_datasets" / "dataset3_filled.csv", index = False)