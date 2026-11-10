"""
Heart Disease Detection - model training and evaluation
Dataset expected: heart.csv (303-row Cleveland-style dataset)
Target column: target (0 = no disease, 1 = disease)
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("heart.csv")
MODEL_PATH = Path("heart_disease_model.pkl")
TARGET = "target"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "heart.csv not found. Place the Cleveland-style heart dataset "
            "in this project folder. See README.md for the expected columns."
        )

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError("The dataset must contain a 'target' column.")

    # Remove rows without a known target and separate features from labels.
    data = data.dropna(subset=[TARGET]).copy()
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    if set(y.unique()) - {0, 1}:
        raise ValueError("Target must use 0 (no disease) and 1 (disease).")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    other_columns = [c for c in X.columns if c not in numeric_columns]

    # Impute missing values. Scale numeric features for Logistic Regression.
    numeric_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    other_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocess_lr = ColumnTransformer([
        ("numeric", numeric_lr, numeric_columns),
        ("other", other_lr, other_columns)
    ])

    numeric_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    other_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocess_tree = ColumnTransformer([
        ("numeric", numeric_tree, numeric_columns),
        ("other", other_tree, other_columns)
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocess_lr),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", preprocess_tree),
            ("classifier", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocess_tree),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=42
            ))
        ])
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    best_name = None
    best_model = None
    best_accuracy = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification report:")
        print(classification_report(y_test, predictions, zero_division=0))

        if accuracy > best_accuracy:
            best_name = name
            best_model = model
            best_accuracy = accuracy

    # Save the best-performing model and its expected input column order.
    joblib.dump(
        {"model": best_model, "features": X.columns.tolist(), "model_name": best_name},
        MODEL_PATH
    )
    print(f"\nBest model: {best_name}")
    print(f"Test accuracy: {best_accuracy * 100:.2f}%")
    print(f"Saved model to: {MODEL_PATH}")
    print("\nEducational project only — this model is not a medical diagnosis tool.")


if __name__ == "__main__":
    main()
