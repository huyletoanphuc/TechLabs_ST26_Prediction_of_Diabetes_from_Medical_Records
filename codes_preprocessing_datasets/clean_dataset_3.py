import numpy as np
import pandas as pd
from pathlib import Path

input_file_path = Path(__file__).parent.parent / "original_datasets" / "dataset3.csv"
output_file_path = Path(__file__).parent.parent / "cleaned_datasets" / "dataset3.csv"

df = pd.read_csv(input_file_path)

selected_columns = [
    "gender",
    "age",
    "bmi",
    "blood_glucose_level",
    "diabetes"
]

df = df[selected_columns]

df.rename(columns={
    "gender": "Gender",
    "age": "Age",
    "bmi": "BMI",
    "blood_glucose_level": "Glucose",
    "diabetes": "Outcome"
}, inplace=True)

df = df[
    [
        "Age",
        "Gender",
        "BMI",
        "Glucose",
        "Outcome"
    ]
]
summary = pd.DataFrame({
    "Type": df.dtypes,
    "Missing": df.isnull().sum(),
    "Unique": df.nunique()
})

print(summary)
gender_map = {"Female" : 1, "Male": 2, "Other" : 0 }

df["Gender"] = df["Gender"].map(gender_map)

df = df[df["Age"] >= 1]
df = df[df["BMI"] < 70]

print(df.isnull().sum())
print(df.describe())

df.to_csv(output_file_path, index=False)