# Water Quality Prediction & Contaminant Early Warning System

A lightweight, low-dependency Machine Learning framework designed to predict Dissolved Oxygen (DO) levels and provide automated early warning risk diagnostics based on continuous water quality parameters[cite: 6].

---

## 📌 Project Overview
Monitoring Dissolved Oxygen (DO) is vital for assessing aquatic ecosystem health and identifying chemical or organic contamination early. This project provides an end-to-end Python pipeline using a **Random Forest Regressor** to model non-linear relationships between standard water parameters (Temperature, pH, Turbidity, Conductance) and Dissolved Oxygen levels[cite: 5, 6].

The pipeline generates regression metrics, classification evaluation indicators, and a visual actual vs. predicted plot, along with a live interactive command-line interface for real-time testing[cite: 4, 6].

---

## 🚀 Key Features
* **Automated Machine Learning Pipeline (`src/data_preprocessing.py`)**: Handles data generation, preprocessing, Random Forest training, visualization export, and summary metric logging[cite: 4, 5].
* **Real-Time CLI Inference Tool (`src/predict.py`)**: Interactive command-line utility allowing users to enter custom sensor readings to predict DO values and receive instant contamination risk alerts[cite: 4, 6].
* **Multi-Tier Risk Assessment Engine**: Automatically classifies predicted DO values into three ecological safety categories:
  * **Safe / Good**: $\text{DO} \ge 7.0 \text{ mg/L}$
  * **Moderate / Caution**: $5.0 \le \text{DO} < 7.0 \text{ mg/L}$
  * **Critical / Hypoxic**: $\text{DO} < 5.0 \text{ mg/L}$[cite: 4]
* **Clean & Low-Dependency Setup**: Built using standard Data Science libraries (`scikit-learn`, `pandas`, `numpy`, `matplotlib`)[cite: 5, 6].

---

## 📊 Performance Summary
* **Regression Metrics**: Evaluated on $R^2$ Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)[cite: 4].
* **Classification Metrics**: Evaluated on Accuracy, Precision, Recall, and F1-Score for risk category assignment[cite: 4].
* **Outputs**: Model outputs are saved under `data/processed/` including the trained model (`water_model.pkl`), visual plots (`prediction_results.png`), and execution reports (`result_summary.txt`)[cite: 4, 6].

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/bhoomi907/water-quality-prediction.git](https://github.com/bhoomi907/water-quality-prediction.git)
   cd water-quality-prediction