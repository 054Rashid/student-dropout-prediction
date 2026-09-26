"""
Student Dropout Prediction - model training.

Based on the original project notebook. The workflow:
1. Loads the student dataset.
2. Encodes the target.
3. Removes the Enrolled class for binary dropout prediction.
4. Standardizes five selected academic/financial features.
5. Splits the data into train/test sets.
6. Trains a Random Forest classifier.
7. Saves the trained model and scaler.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/Dataset.csv"

FEATURES = [
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

def prepare_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df["Target"] = LabelEncoder().fit_transform(df["Target"])

    # Remove the Enrolled class and create a binary Dropout target.
    df = df[df["Target"] != 1].copy()
    df["Dropout"] = df["Target"].apply(lambda x: 1 if x == 0 else 0)

    x = df[FEATURES].values
    y = df["Dropout"].values

    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    return x, y, scaler

def train():
    x, y, scaler = prepare_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    model = RandomForestClassifier(
        n_estimators=500,
        criterion="entropy"
    )
    model.fit(x_train, y_train)

    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train()
