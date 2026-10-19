import pandas as pd
from pathlib import Path

flagCheckAge = False
flagCheckBMI = False
flagCheckBloodPressure = False
flagCheckGlucose = False
flagCheckInsulin = False
flagCheckOutcome = False

input_file_path = Path(__file__).parent.parent / "original_datasets" / "dataset1.csv"
output_file_path = Path(__file__).parent.parent / "cleaned_datasets" / "dataset1.csv"

df = pd.read_csv(input_file_path)

df["Gender"] = 1

df = df[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Glucose",
        "Insulin",
        "Outcome"
    ]
]

# Replace BMI = 0 with median of non-zero BMI values
bmi_median_nonzero = df.loc[df["BMI"] != 0, "BMI"].median()
df.loc[df["BMI"] == 0, "BMI"] = bmi_median_nonzero

# Replace BloodPressure = 0 with median of non-zero BloodPressure values
bp_median_nonzero = df.loc[df["BloodPressure"] != 0, "BloodPressure"].median()
df.loc[df["BloodPressure"] == 0, "BloodPressure"] = bp_median_nonzero

# Replace Glucose = 0 with median of non-zero Glucose values
glucose_median_nonzero = df.loc[df["Glucose"] != 0, "Glucose"].median()
df.loc[df["Glucose"] == 0, "Glucose"] = glucose_median_nonzero

# Replace Insulin = 0 with median of non-zero Insulin values
insulin_median_nonzero = df.loc[df["Insulin"] != 0, "Insulin"].median()
df.loc[df["Insulin"] == 0, "Insulin"] = insulin_median_nonzero

df.to_csv(output_file_path, index = False)

if flagCheckAge:
    age_numeric = pd.to_numeric(df["Age"], errors="coerce")

    not_numeric_age = df[age_numeric.isna()]

    not_integer_age = df[
        age_numeric.notna() & (age_numeric % 1 != 0)
    ]

    print("\n========== AGE CHECK ==========")

    print("\nNon-numeric Age values:")
    print(not_numeric_age)

    print("\nNon-integer Age values:")
    print(not_integer_age)

    print("\nMinimum Age:")
    print(age_numeric.min())

    print("\nMaximum Age:")
    print(age_numeric.max())
    
if flagCheckBMI:
    bmi_numeric = pd.to_numeric(df["BMI"], errors="coerce")

    print("\n========== BMI CHECK ==========")

    print("\nMinimum BMI:")
    print(bmi_numeric.min())

    print("\nMaximum BMI:")
    print(bmi_numeric.max())

    bmi_zero = df[df["BMI"] == 0]

    print("\nRows with BMI = 0:")
    print(bmi_zero)

    print("\nNumber of rows with BMI = 0:")
    print(len(bmi_zero))

    bmi_very_high = df[df["BMI"] > 60]

    print("\nRows with BMI > 60:")
    print(bmi_very_high)

    print("\nNumber of rows with BMI > 60:")
    print(len(bmi_very_high))
    
if flagCheckBloodPressure:
    print("\n========== BLOOD PRESSURE CHECK ==========")

    bp_numeric = pd.to_numeric(df["BloodPressure"], errors="coerce")

    print("Minimum BloodPressure:", bp_numeric.min())
    print("Maximum BloodPressure:", bp_numeric.max())
    print("Number of BloodPressure = 0:", (bp_numeric == 0).sum())
    print("Number of negative BloodPressure:", (bp_numeric < 0).sum())
    
    BP_zero = df[df["BloodPressure"] == 0]

    print("\nRows with BP = 0:")
    print(BP_zero)

    print("\nNumber of rows with BP = 0:")
    print(len(BP_zero))
    
if flagCheckGlucose:
    print("\n========== GLUCOSE CHECK ==========")

    glucose_numeric = pd.to_numeric(df["Glucose"], errors="coerce")

    print("Minimum Glucose:", glucose_numeric.min())
    print("Maximum Glucose:", glucose_numeric.max())
    print("Number of Glucose = 0:", (glucose_numeric == 0).sum())
    print("Number of negative Glucose:", (glucose_numeric < 0).sum())
    
    glucose_zero = df[df["Glucose"] == 0]

    print("\nRows with Glucose = 0:")
    print(glucose_zero)

    print("\nNumber of rows with Glucose = 0:")
    print(len(glucose_zero))
    
if flagCheckInsulin:
    print("\n========== INSULIN CHECK ==========")

    insulin_numeric = pd.to_numeric(df["Insulin"], errors="coerce")

    print("Minimum Insulin:", insulin_numeric.min())
    print("Maximum Insulin:", insulin_numeric.max())
    print("Number of Insulin = 0:", (insulin_numeric == 0).sum())
    print("Number of negative Insulin:", (insulin_numeric < 0).sum())
    
    insulin_zero = df[df["Insulin"] == 0]

    print("\nRows with Insulin = 0:")
    print(insulin_zero)

    print("\nNumber of rows with Insulin = 0:")
    print(len(insulin_zero))
    
    insulin_very_high = df[df["Insulin"] > 600]

    print("\nRows with Insulin > 600:")
    print(insulin_very_high)

    print("\nNumber of rows with Insulin > 600:")
    print(len(insulin_very_high))

if flagCheckOutcome:   
    print("\n========== OUTCOME CHECK ==========")

    outcome_numeric = pd.to_numeric(df["Outcome"], errors="coerce")

    # Non-numeric values
    non_numeric_outcome = df[outcome_numeric.isna()]

    # Numeric but not integer
    non_integer_outcome = df[
        outcome_numeric.notna() & (outcome_numeric % 1 != 0)
    ]

    # Values not 0 or 1
    invalid_outcome = df[
        outcome_numeric.notna() & ~outcome_numeric.isin([0, 1])
    ]

    print("Unique Outcome values:")
    print(df["Outcome"].unique())

    print("\nMinimum Outcome:")
    print(outcome_numeric.min())

    print("\nMaximum Outcome:")
    print(outcome_numeric.max())

    print("\nNumber of non-numeric Outcome values:")
    print(len(non_numeric_outcome))

    print("\nNumber of non-integer Outcome values:")
    print(len(non_integer_outcome))

    print("\nNumber of invalid Outcome values, not 0 or 1:")
    print(len(invalid_outcome))

    print("\nRows with non-numeric Outcome:")
    print(non_numeric_outcome)

    print("\nRows with non-integer Outcome:")
    print(non_integer_outcome)

    print("\nRows with invalid Outcome values:")
    print(invalid_outcome)