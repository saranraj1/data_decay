import numpy as np
import pandas as pd
from scipy import stats

class DriftDetector:
    """
    Research-grade suite for detecting statistical decay and 
    distributional shifts in production machine learning models.
    """
    
    @staticmethod
    def calculate_psi(expected, actual, buckets=10):
        """
        Calculates the Population Stability Index (PSI).
        PSI measures how much the distribution of a variable has shifted 
        between two points in time (e.g., Training vs. Production).
        
        PSI Interpretation:
        - PSI < 0.1: No significant change (Model is stable).
        - 0.1 <= PSI < 0.2: Slight shift (Monitor closely).
        - PSI >= 0.2: Significant drift (Action required: Retrain).
        """
        def sub_psi(e_perc, a_perc):
            """Component of the PSI formula: (Actual% - Expected%) * ln(Actual% / Expected%)"""
            if a_perc == 0: a_perc = 0.0001  # Convergence stabilization
            if e_perc == 0: e_perc = 0.0001
            return (a_perc - e_perc) * np.log(a_perc / e_perc)

        # Create buckets based on the 'Expected' (Baseline) distribution
        breakpoints = np.percentile(expected, np.arange(0, 100, 100 / buckets))
        breakpoints = np.unique(breakpoints) # Handle variables with low variance
        
        # Calculate frequencies per bucket
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        psi_value = 0
        for i in range(len(expected_percents)):
            psi_value += sub_psi(expected_percents[i], actual_percents[i])

        return round(psi_value, 4)

    @staticmethod
    def detect_ks_drift(baseline_series, new_series, alpha=0.05):
        """
        Two-sample Kolmogorov-Smirnov test.
        A non-parametric test that compares the cumulative distributions 
        to determine if they originate from the same distribution.
        """
        # Ensure no NaNs are passed to the statistical test
        baseline = baseline_series.dropna()
        new_data = new_series.dropna()
        
        ks_stat, p_val = stats.ks_2samp(baseline, new_data)
        
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": round(ks_stat, 4),
            "p_value": round(p_val, 6),
            "drift_detected": p_val < alpha
        }

    def generate_feature_drift_report(self, train_df, production_df, features):
        """
        Orchestrates a comprehensive drift audit across multiple features.
        """
        report = []
        for feat in features:
            psi = self.calculate_psi(train_df[feat], production_df[feat])
            ks = self.detect_ks_drift(train_df[feat], production_df[feat])
            
            # Hybrid Logic: A feature drifts if PSI is high OR KS test is significant
            status = "DRIFT" if (psi >= 0.2 or ks['drift_detected']) else "STABLE"
            
            report.append({
                "Feature": feat,
                "PSI_Score": psi,
                "KS_P_Value": ks['p_value'],
                "Audit_Status": status
            })
            
        return pd.DataFrame(report)

if __name__ == "__main__":
    print("🔬 Drift Detector Module Initialized.")