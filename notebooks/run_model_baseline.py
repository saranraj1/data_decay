import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import numpy as np

# Add project root to sys.path to access src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import prepare_decay_experiment

def main():
    print("🚀 Starting Model Baseline Execution...")
    
    # Step 1: Initialize the Timeline Split
    # Using the local path for California Housing CSV
    # Since we are in notebooks/, the data is at ../data/raw/housing.csv
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/housing.csv')
    
    # Normalize path
    DATA_PATH = os.path.abspath(DATA_PATH)
    
    print(f"📂 Loading data from: {DATA_PATH}")
    baseline_df, future_df = prepare_decay_experiment(DATA_PATH)
    
    # Define features and target (Predicting median_house_value)
    target = 'median_house_value'
    # Drop categorical for baseline as per notebook
    X_baseline = baseline_df.drop(columns=[target, 'ocean_proximity']) 
    y_baseline = baseline_df[target]

    # Step 2: Model Training
    print("🧠 Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_baseline, y_baseline)
    
    # Internal Validation (Sanity Check)
    y_pred = model.predict(X_baseline)
    mse = mean_squared_error(y_baseline, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_baseline, y_pred)

    
    print(f"✅ Baseline Training Complete.")
    print(f" - RMSE: ${rmse:,.2f}")
    print(f" - R2 Score: {r2:.4f}")

    # Step 3: Distribution Profiling
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_pred, label='Baseline Predictions', color='blue', lw=2)
    plt.title("Baseline Prediction Distribution (Ground Truth)", fontsize=14)
    plt.xlabel("Predicted House Value")
    
    # Ensure reports/figures exists
    figures_dir = os.path.join(os.path.dirname(__file__), '../reports/figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    save_path = os.path.join(figures_dir, 'baseline_prediction_dist.png')
    plt.savefig(save_path)
    print(f"📊 Distribution plot saved to: {save_path}")
    # plt.show() # Cannot show in headless mode
    
    # Store Feature Importance for future Comparison
    importance_df = pd.DataFrame({
        'Feature': X_baseline.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importances:")
    print(importance_df.head())

    # Step 4: Persistence
    models_dir = os.path.join(os.path.dirname(__file__), '../models/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, 'golden_model.pkl')
    feature_list_path = os.path.join(models_dir, 'feature_list.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(X_baseline.columns.tolist(), feature_list_path)
    
    print(f"💾 Model and Stability Profile serialized to {models_dir}")

if __name__ == "__main__":
    main()
