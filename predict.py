"""
Interactive prediction using the trained heart disease model.
Run train_model.py first.
"""

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("heart_disease_model.pkl")


def main():
    if not MODEL_PATH.exists():
        print("Model file not found. Run: python train_model.py")
        return

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    print(f"Model: {saved['model_name']}")
    print("\nEnter patient feature values.")
    print("Use the same feature names/order as the columns in heart.csv.")
    values = {}

    for feature in features:
        while True:
            raw = input(f"{feature}: ").strip()
            try:
                values[feature] = float(raw)
                break
            except ValueError:
                print("Please enter a numeric value.")

    input_data = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])
        print(f"\nModel output: {prediction}")
        print(f"Estimated class-1 probability: {probability * 100:.1f}%")

    if prediction == 1:
        print("The model classified this sample as class 1 (heart disease).")
    else:
        print("The model classified this sample as class 0 (no heart disease).")

    print("This is an educational ML demonstration, not a diagnosis. "
          "Consult a qualified healthcare professional for medical concerns.")


if __name__ == "__main__":
    main()
