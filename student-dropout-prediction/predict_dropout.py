"""
Student Dropout Prediction - command-line predictor.

Loads the saved Random Forest model and StandardScaler and predicts
Dropout or Non-Dropout from five input features.
"""

import joblib
import numpy as np

MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_dropout():
    print("Please enter the following details:")

    tuition_fees = float(input("Tuition fees up to date (0 or 1): "))
    curricular_units_1st_approved = int(
        input("Curricular units 1st sem (approved) (0-10): ")
    )
    curricular_units_1st_grade = int(
        input("Curricular units 1st sem (grade) (0-20): ")
    )
    curricular_units_2nd_approved = int(
        input("Curricular units 2nd sem (approved) (0-10): ")
    )
    curricular_units_2nd_grade = int(
        input("Curricular units 2nd sem (grade) (0-20): ")
    )

    input_data = [
        tuition_fees,
        curricular_units_1st_approved,
        curricular_units_1st_grade,
        curricular_units_2nd_approved,
        curricular_units_2nd_grade,
    ]

    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    result = "Dropout" if prediction[0] == 1 else "Non-Dropout"

    print(f"The prediction for the provided data is: {result}")

if __name__ == "__main__":
    predict_dropout()
