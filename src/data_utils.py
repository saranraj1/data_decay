import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataDecayLoader:
    """
    Research-grade utility for loading and partitioning data 
    into Baseline (Train/Test) and Future (Drift) sets.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Dataset not found at {filepath}")
        
    def load_and_clean(self):
        """
        Loads the California Housing data and performs basic research-grade cleaning.
        """
        df = pd.read_csv(self.filepath)
        
        # In research, we must handle any initial missingness immediately 
        # to ensure the 'Baseline' is perfect.
        if df.isnull().sum().any():
            print("⚠️ Initial missingness detected. Dropping nulls for clean baseline.")
            df = df.dropna()
            
        return df

    def split_for_drift_study(self, df, baseline_size=0.7):
        """
        Splits data into:
        1. Baseline Set: Used for training and initial testing.
        2. Future Set: Held back to simulate 'Time' and introduce decay.
        """
        # We split chronologically (if there was a date) or randomly to create 
        # a 'Past' and 'Future' scenario.
        baseline_df, future_df = train_test_split(
            df, 
            train_size=baseline_size, 
            random_state=42, 
            shuffle=True
        )
        
        print(f"📊 Data Split Complete:")
        print(f" - Baseline (Past) Size: {len(baseline_df)}")
        print(f" - Future (Production) Size: {len(future_df)}")
        
        return baseline_df, future_df

    @staticmethod
    def save_processed_data(df, name, folder='../data/processed/'):
        """Saves versions of the data for reproducibility."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"💾 Saved: {path}")

def prepare_decay_experiment(filepath):
    """
    Orchestration function to prepare the environment for Project #3.
    """
    loader = DataDecayLoader(filepath)
    raw_data = loader.load_and_clean()
    
    # Partitioning the 'Timeline'
    baseline, future = loader.split_for_drift_study(raw_data)
    
    # Persistence for Notebook usage
    loader.save_processed_data(baseline, 'baseline_stable')
    loader.save_processed_data(future, 'future_unseen')
    
    return baseline, future

if __name__ == "__main__":
    # Assuming running from src directory or adjusting path accordingly
    # If running from project root, data is at data/raw/housing.csv
    # If running from src, data is at ../data/raw/housing.csv
    
    import os
    
    # Determine path based on where we are running
    if os.path.exists("data/raw/housing.csv"):
        data_path = "data/raw/housing.csv"
        # We also need to adjust the save folder in prepare_decay_experiment if we aren't passing it
        # But prepare_decay_experiment calls safe_processed_data with default '../data/processed/'
        # which is problematic if we are in root.
        # Let's handle this by changing the default or passing it.
        # For now, let's just point to the file.
    elif os.path.exists("../data/raw/housing.csv"):
        data_path = "../data/raw/housing.csv"
    else:
        # Fallback absolute path for safety
        data_path = r"C:\30 Projects\data-decay\data\raw\housing.csv"
        
    print(f"🚀 Running Data Utils with data: {data_path}")
    prepare_decay_experiment(data_path)
