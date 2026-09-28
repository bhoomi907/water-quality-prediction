import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def classify_water_quality(do_val):
    if do_val >= 7.0:
        return "Safe / Good"
    elif 5.0 <= do_val < 7.0:
        return "Moderate / Caution"
    else:
        return "Critical / Hypoxic"

def run_water_quality_pipeline():
    np.random.seed(42)
    n_samples = 1000
    
    # 1. Generate synthetic water data
    timestamps = pd.date_range(start="2026-01-01", periods=n_samples, freq="h")
    temp = 18 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 0.5, n_samples)
    ph = 7.5 + 0.4 * np.cos(2 * np.pi * np.arange(n_samples) / 24) + np.random.normal(0, 0.1, n_samples)
    turbidity = 5.0 + np.random.exponential(scale=1.5, size=n_samples)
    conductance = 400 + 50 * np.sin(2 * np.pi * np.arange(n_samples) / 168) + np.random.normal(0, 10, n_samples)
    
    # Dissolved Oxygen target variable
    do = 12 - 0.25 * temp - 0.1 * turbidity + np.random.normal(0, 0.2, n_samples)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Temperature": temp,
        "pH": ph,
        "Turbidity": turbidity,
        "Conductance": conductance,
        "Dissolved_Oxygen": do
    })
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/water_quality_dataset.csv", index=False)
    
    # 2. Train/Test Split
    X = df.drop(columns=["Timestamp", "Dissolved_Oxygen"])
    y = df["Dissolved_Oxygen"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "data/processed/water_model.pkl")
    
    # 4. Predictions and Regression Metrics
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 5. Categorical Risk Classification Metrics
    results_df = pd.DataFrame({
        "Actual_DO": y_test.values,
        "Predicted_DO": predictions
    })
    
    results_df["Actual_Status"] = results_df["Actual_DO"].apply(classify_water_quality)
    results_df["Predicted_Status"] = results_df["Predicted_DO"].apply(classify_water_quality)
    
    # Calculate paper evaluation metrics
    acc = accuracy_score(results_df["Actual_Status"], results_df["Predicted_Status"])
    prec = precision_score(results_df["Actual_Status"], results_df["Predicted_Status"], average='weighted', zero_division=0)
    rec = recall_score(results_df["Actual_Status"], results_df["Predicted_Status"], average='weighted', zero_division=0)
    f1 = f1_score(results_df["Actual_Status"], results_df["Predicted_Status"], average='weighted', zero_division=0)
    
    status_counts = results_df["Predicted_Status"].value_counts()
    
    # 6. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual Dissolved Oxygen", color="blue", linewidth=1.5)
    plt.plot(predictions[:100], label="Predicted Dissolved Oxygen", color="orange", linestyle="--", linewidth=1.5)
    plt.title("Water Quality Prediction: Actual vs Predicted (First 100 Test Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Dissolved Oxygen (mg/L)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("data/processed/prediction_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Print Terminal Text Summary Report
    summary_text = f"""
=========================================================
            WATER QUALITY ANALYSIS SUMMARY REPORT        
=========================================================
[1] REGRESSION ACCURACY METRICS:
    - Model Accuracy (R2 Score) : {r2 * 100:.2f}%
    - Root Mean Squared Error   : {rmse:.4f} mg/L
    - Mean Absolute Error       : {mae:.4f} mg/L

[2] CLASSIFICATION EVALUATION METRICS:
    - Accuracy  : {acc * 100:.2f}%
    - Precision : {prec:.4f}
    - Recall    : {rec:.4f}
    - F1-Score  : {f1:.4f}

[3] WATER QUALITY RISK DISTRIBUTION (Predicted Test Samples):
    - Safe / Good (DO >= 7.0)     : {status_counts.get('Safe / Good', 0)} samples
    - Moderate / Caution (5-7)   : {status_counts.get('Moderate / Caution', 0)} samples
    - Critical / Hypoxic (< 5.0)  : {status_counts.get('Critical / Hypoxic', 0)} samples
=========================================================
"""
    print(summary_text)
    
    with open("data/processed/result_summary.txt", "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    run_water_quality_pipeline()