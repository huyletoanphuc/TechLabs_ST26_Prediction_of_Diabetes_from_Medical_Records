import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent

df1 = pd.read_csv(script_dir.parent / "cleaned_datasets" / "dataset1.csv")
df2 = pd.read_csv(script_dir.parent / "cleaned_datasets" / "dataset2_filled.csv")
df3 = pd.read_csv(script_dir.parent / "cleaned_datasets" / "dataset3_filled.csv")
df4 = pd.read_csv(script_dir.parent / "cleaned_datasets" / "dataset4.csv")

df1 = df1[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Insulin",
        "Outcome",
    ]
]

df2 = df2[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Insulin",
        "Outcome",
    ]
]

df3 = df3[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Insulin",
        "Outcome",
    ]
]

df4 = df4[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Insulin",
        "Outcome",
    ]
]

# ==========================
# Combine all datasets
# ==========================
combined_df = pd.concat(
    [df1, df2, df3, df4],
    axis=0,
    ignore_index=True
)

combined_df.to_csv(script_dir / "combined_dataset.csv", index=False)

print("Combined dataset saved as 'combined_dataset.csv'")
print("Shape:", combined_df.shape)
print(combined_df.columns.tolist())