import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import heapq
import random
from pathlib import Path

input_file_path = Path(__file__).parent.parent / "original_datasets" / "dataset2.csv"
output_file_path = Path(__file__).parent.parent / "cleaned_datasets" / "dataset2.csv"

diabetes_df = pd.read_csv(input_file_path)

diabetes_df.rename(columns={
    "BPLevel": "BloodPressure",
    "Diabetic": "Outcome"
}, inplace=True)

diabetes_df = diabetes_df[
    [
        "Age",
        "Gender",
        "BMI",
        "BloodPressure",
        "Outcome"
    ]
]

def random_age(age):
    if age == 'less than 40':
        return random.randint(0, 40)
    elif age == '60 or older':
        return random.randint(60, 100)
    elif age == '40-49':
        return random.randint(40, 49)
    elif age == '50-59':
        return random.randint(50, 59)
    return age

def random_bp(bp):
    # assumption/suggestion (according to guidelines of the European society for hypertension):
    # only use the systolic blood pressure
    # high-blood pressue: > 140mmHg
    # normal: 120-140mmHg
    # low: <120mmHg
    if bp == 'high' or "High":
        return random.randint(140, 180)
    elif bp == 'normal' or "normal ":
        return random.randint(120,140)
    elif bp == 'low' or "Low":
        return random.randint(100,120)
    return bp

diabetes_df['Age'] = diabetes_df['Age'].apply(random_age)
diabetes_df['BloodPressure'] = diabetes_df['BloodPressure'].apply(random_bp)

# male 2, female 1
diabetes_df['Gender'].replace('Male','2', inplace=True)
diabetes_df['Gender'].replace('Female','1', inplace=True)

# output 1 = yes
# 0 = no
# na = no = 0
diabetes_df['Outcome'] = diabetes_df['Outcome'].fillna(0)
diabetes_df['Outcome'].replace('yes','1', inplace=True)
diabetes_df['Outcome'].replace(' no','0', inplace=True)
diabetes_df['Outcome'].replace('no','0', inplace=True)

diabetes_df.to_csv(output_file_path, index=False)