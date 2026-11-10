# Heart Disease Detection Using Machine Learning

A beginner-friendly machine-learning project that compares Logistic Regression,
Decision Tree, and Random Forest classifiers using patient health measurements.

## Project objective

The project explores whether a model can classify records into:
- `0`: no heart disease
- `1`: heart disease

This is an educational project, not a clinical diagnostic system.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib (can be used for additional visualizations)

## Dataset

Place a Cleveland-style heart disease CSV file named `heart.csv` in this folder.
It should contain a binary `target` column and numeric feature columns. The
commonly used 303-record Cleveland dataset has these feature names:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

The training script uses the feature columns present in your CSV, so the target
must be named `target`. Do not upload private patient information.

## Setup

```bash
python -m venv .venv
```

Activate the environment:

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install packages:
```bash
pip install -r requirements.txt
```

## Train and compare models

```bash
python train_model.py
```

The script splits the data into training and test sets, preprocesses missing
values, compares three classifiers, prints accuracy/confusion matrices/
classification reports, and saves the best model as
`heart_disease_model.pkl`.

Accuracy depends on the exact dataset and train/test split. Do not claim a
specific accuracy unless it is the result produced by your own run.

## Test with entered values

After training:

```bash
python predict.py
```

Enter one numeric value for each feature shown by the program.

## Project workflow

1. Load the dataset with Pandas.
2. Separate input features and the target.
3. Handle missing values.
4. Split the data into training and testing sets.
5. Train Logistic Regression, Decision Tree, and Random Forest.
6. Compare evaluation results.
7. Save the best model.
8. Enter sample feature values and generate a model prediction.

## Important note

This project is for learning and demonstration only. It must not be used to
diagnose, treat, or make medical decisions about anyone.
