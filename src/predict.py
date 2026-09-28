import joblib
import pandas as pd

def predict_water_quality():
    model_path = "data/processed/water_model.pkl"
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Trained model file not found! Please run data_preprocessing.py first.")
        return

    print("==================================================")
    print("      WATER QUALITY EARLY WARNING INFERENCE       ")
    print("==================================================")
    print("Enter real-time sensor parameters to predict Dissolved Oxygen:\n")

    try:
        temp = float(input("Enter Water Temperature (°C) [e.g., 20.5]: "))
        ph = float(input("Enter pH Level [e.g., 7.4]: "))
        turbidity = float(input("Enter Turbidity (NTU) [e.g., 4.2]: "))
        conductance = float(input("Enter Specific Conductance (µS/cm) [e.g., 410]: "))
    except ValueError:
        print("\nInvalid input! Please enter numeric values only.")
        return

    # Prepare input for model
    input_data = pd.DataFrame([{
        "Temperature": temp,
        "pH": ph,
        "Turbidity": turbidity,
        "Conductance": conductance
    }])

    # Generate prediction
    predicted_do = model.predict(input_data)[0]

    # Classify safety level
    if predicted_do >= 7.0:
        status = "SAFE / GOOD"
        alert = "Water quality is healthy. Aquatic life supported."
    elif 5.0 <= predicted_do < 7.0:
        status = "MODERATE / CAUTION"
        alert = "Elevated risk. Monitor contaminant runoff closely."
    else:
        status = "CRITICAL / HYPOXIC ALERT"
        alert = "WARNING: Low oxygen levels! Immediate intervention required."

    print("\n--------------------------------------------------")
    print(f"PREDICTED DISSOLVED OXYGEN : {predicted_do:.2f} mg/L")
    print(f"WATER SAFETY STATUS        : {status}")
    print(f"EARLY WARNING ACTION       : {alert}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    predict_water_quality()