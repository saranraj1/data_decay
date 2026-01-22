# Data Decay & Drift Detection (Project: Data Decay)

## 📌 Project Overview
**Data Decay** is a research-oriented project designed to simulate, detect, and analyze **Distribution Shift** (Data Drift) and **Concept Drift** in machine learning production environments. 

Using the **California Housing** dataset, this project establishes a baseline predictive model, simulates real-world decay scenarios (such as sensor failure and economic inflation), and utilizes statistical tests to audit the model's stability over time.

## 🏗 Project Structure

```
data-decay/
├── data/
│   ├── raw/                # Original California Housing dataset
│   └── processed/          # Simulated drift datasets (future_sensor_decay, future_concept_drift)
├── models/                 # Serialized models (.pkl) & feature lists
├── notebooks/
│   ├── model_baseline.ipynb   # Trains the "Golden" baseline model & establishes ground truth
│   ├── decay_simulation.ipynb # Injects synthetic noise (Sensor Drift) and shifts (Concept Drift)
│   └── detection_audit.ipynb  # Runs the DriftDetector suite (PSI, KS-Test) against decayed data
├── reports/
│   └── figures/            # Generated drift visualizations (CDF plots, etc.)
├── src/
│   ├── data_utils.py       # Utilities for loading, cleaning, and partitioning data
│   └── drift_detector.py   # Core logic for PSI and Kolmogorov-Smirnov statistical tests
└── README.md
```

## ⚙️ Installation & Setup

1. **Clone or Navigate to the Project Directory**:
   ```bash
   cd "C:/30 Projects/data-decay"
   ```

2. **Install Dependencies**:
   Ensure you have the required Python packages. You can install them via pip (requirements file is not provided, but standard datascience stack is used):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
   ```
   *(Note: The `src` folder is a local package. Ensure your PYTHONPATH is set correctly or run from the project root.)*

## 🚀 Usage Guide

This project is executed in a sequential 3-step workflow using Jupyter Notebooks.

### Step 1: Establish Baseline (`model_baseline.ipynb`)
- Loads raw data and splits it into a **Baseline (Past)** and **Future (Unseen)** set.
- Trains a **Random Forest Regressor** on the Baseline data.
- Serializes the "Golden Model" for future comparisons.
- **Output**: `models/golden_model.pkl`, `data/processed/baseline_stable.csv`

### Step 2: Simulate Decay (`decay_simulation.ipynb`)
- Loads the held-out **Future** data.
- Injects two specific types of drift:
    - **feature_drift (Sensor Noise)**: Adds Gaussian noise to `housing_median_age`.
    - **concept_drift (Economic Inflation)**: Shift in `median_income` distribution.
- **Output**: `data/processed/future_sensor_decay.csv`, `data/processed/future_concept_drift.csv`

### Step 3: Run Detection Audit (`detection_audit.ipynb`)
- Loads the Baseline Production data and the Decayed data.
- Uses the `DriftDetector` class to perform:
    - **PSI (Population Stability Index)**: To measure magnitude of shift.
    - **KS Test (Kolmogorov-Smirnov)**: To detect statistical distribution changes.
- Generates alerts if potential drift is found (PSI > 0.2 or p-value < 0.05).
- Detects the degradation in model accuracy (RMSE increase).
- **Output**: Drift Reports and CDF Visualizations in `reports/figures`.

## 🧪 Methodology: The DriftDetector

The core logic resides in `src/drift_detector.py`. It implements a hybrid detection strategy:

1.  **Population Stability Index (PSI)**:
    - Measures the shift in distribution buckets.
    - **Thresholds**:
        - PSI < 0.1: Stable
        - 0.1 ≤ PSI < 0.2: Slight Shift
        - PSI ≥ 0.2: **Critical Drift**

2.  **Kolmogorov-Smirnov (KS) Test**:
    - A non-parametric test comparing the cumulative distribution functions (CDF) of the baseline and production data.
    - **Trigger**: A p-value < 0.05 indicates the samples come from different distributions.

3.  **Visualization**:
    - Visualizes drift using **Cumulative Distribution Function (CDF)** plots to show exactly where the divergence occurs.

## 📊 Key Results
- **Sensor Drift Scenario**: Successfully detects noise injection in `housing_median_age`.
- **Concept Drift Scenario**: Successfully detects inflation impact on `median_income` and correlates it with a drop in model RMSE accuracy.

---

