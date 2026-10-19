import numpy as np
import pandas as pd
from pathlib import Path

input_file_path = Path(__file__).parent.parent / "original_datasets" / "dataset4.csv"
output_file_path = Path(__file__).parent.parent / "cleaned_datasets" / "dataset4.csv"

tab4 = pd.read_csv(input_file_path)
tab4.insert(1, "Gender", 1)

new_order = ['Age', "Gender", "BMI", "BloodPressure", "Glucose", "Insulin", "Outcome"]
tab44 = tab4.reindex(columns= new_order)

tab44.to_csv(output_file_path, index = False)