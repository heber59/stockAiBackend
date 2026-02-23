import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.models.stock_model import StockModel

def analyze():
    # 1. Load Global Model
    model_dir = "models"
    model = StockModel()
    model.load(model_dir, "xgboost_global_stock_model")
    
    # 2. Re-create Pooled Test Set (matching ModelTrainer logic)
    features_dir = "data/features"
    symbols = ["AAPL", "MSFT", "VRT", "AVGO"]
    all_test_dfs = []
    
    for s in symbols:
        df = pd.read_parquet(os.path.join(features_dir, f"{s}.parquet"))
        split_idx = int(len(df) * 0.8)
        all_test_dfs.append(df.iloc[split_idx:])
        
    test_df = pd.concat(all_test_dfs)
    X_test = test_df[model.FEATURES]
    y_test = test_df[model.TARGET]
    
    # 3. Get raw probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []
    
    print(f"\n{'='*60}")
    print(f"{'Threshold':<12} | {'Precision':<12} | {'Signals Count':<15} | {'% of Test Set'}")
    print(f"{'-'*60}")
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        # Calculate precision ONLY if there are signals
        if np.sum(y_pred) > 0:
            precision = precision_score(y_test, y_pred, zero_division=0)
            count = np.sum(y_pred)
            percentage = (count / len(y_test)) * 100
            print(f"{t:<12.2f} | {precision:<12.4f} | {count:<15} | {percentage:.1f}%")
        else:
            print(f"{t:<12.2f} | {'N/A':<12} | {'0':<15} | 0.0%")

if __name__ == "__main__":
    analyze()
