# Student Dropout Prediction

A machine-learning classification project for predicting whether a student is likely to **drop out** or **not drop out** based on selected academic and tuition-related features.

The project includes exploratory data analysis, preprocessing, comparison of multiple classification algorithms, Random Forest model training, evaluation, model persistence, and a command-line prediction script.

## Project Overview

The original notebook works with a student dataset containing **4,424 records and 37 columns**.

The original target has three classes:

- Dropout
- Enrolled
- Graduate

For the binary dropout prediction task, the **Enrolled** class is removed and the remaining records are mapped to:

- `1` → Dropout
- `0` → Non-Dropout

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Random Forest
- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Classifier
- Perceptron
- K-Nearest Neighbors

## Project Workflow

1. Load the student dataset.
2. Inspect data types, missing values, descriptive statistics, and target distribution.
3. Encode the target variable.
4. Remove the Enrolled class for binary dropout prediction.
5. Create the binary `Dropout` target.
6. Select five features.
7. Standardize the selected features using `StandardScaler`.
8. Split the data into training and test sets using an 80/20 split.
9. Train and evaluate multiple classification models.
10. Compare model performance.
11. Train the Random Forest model used for the saved predictor.
12. Save the Random Forest model and scaler.
13. Use the saved artifacts for command-line predictions.

## Features Used

The final Random Forest predictor uses these five features:

- Tuition fees up to date
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)

## Models Compared

The notebook evaluates:

1. Gaussian Naive Bayes
2. Logistic Regression
3. Random Forest
4. Support Vector Classifier
5. Perceptron
6. K-Nearest Neighbors

The notebook also evaluates different K values for KNN before using `n_neighbors=3` for the final KNN evaluation.

## Model Results

The original notebook uses an 80/20 train-test split with `random_state=1`.

| Model | Accuracy |
|---|---:|
| Gaussian Naive Bayes | 84.71% |
| Logistic Regression | 88.71% |
| Random Forest | **88.98%** |
| Support Vector Classifier | 88.02% |
| Perceptron | 82.09% |
| K-Nearest Neighbors | 87.74% |

### Random Forest Classification Report

The notebook reports:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Non-Dropout | 0.91 | 0.92 | 0.91 |
| Dropout | 0.86 | 0.85 | 0.86 |
| Overall accuracy | | | **0.89** |

The Random Forest model was trained with:

- `n_estimators=500`
- `criterion="entropy"`

## Prediction Script

The repository includes `predict_dropout.py`, which loads:

```text
models/random_forest_model.pkl
models/scaler.pkl
```

and accepts the five model features interactively.

Example from the original notebook:

```text
Tuition fees up to date: 1
1st semester approved: 10
1st semester grade: 20
2nd semester approved: 10
2nd semester grade: 20
```

The recorded prediction for this example was:

```text
Non-Dropout
```

## Repository Structure

```text
student-dropout-prediction/
├── README.md
├── requirements.txt
├── train_model.py
├── predict_dropout.py
├── .gitignore
│
├── data/
│   └── Dataset.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Dropout_Prediction.ipynb
│
└── visualizations/
    └── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

From the repository root:

```bash
python train_model.py
```

This trains the Random Forest model and saves:

```text
models/random_forest_model.pkl
models/scaler.pkl
```

## Make a Prediction

```bash
python predict_dropout.py
```

Then enter the requested student information when prompted.

## Reproduce the Full Analysis

For the complete exploratory analysis, model comparison, evaluation, and original workflow, open:

```text
notebooks/Dropout_Prediction.ipynb
```

in Jupyter Notebook or Google Colab.

## Notes

This repository reflects the work implemented in the original project notebook. The saved model and scaler are included as the supplied trained artifacts.

This project is an educational machine-learning project and should not be treated as a standalone academic or institutional decision-making system.

## Author

**Mohammad Rashid**

B.Tech in Artificial Intelligence and Data Science
