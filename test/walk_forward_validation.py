import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, accuracy_score

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.models.stock_model import StockModel

def walk_forward_validation(train_size_rows=2000, test_size_rows=500):
    print(f"\n⏳ STARTING WALK-FORWARD VALIDATION ⏳")
    print(f"Windows: Train={train_size_rows} rows, Test={test_size_rows} rows\n")

    # 1. Load and Pool Data (sorted strictly by date)
    features_dir = "data/features"
    symbols = ["AAPL", "MSFT", "VRT", "AVGO", "SPY", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "QCOM", "NFLX"]
    
    all_dfs = []
    for s in symbols:
        path = os.path.join(features_dir, f"{s}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Add symbol column to keep track if needed, but the main index is Date
            all_dfs.append(df)
            
    # Combine and sort by date index
    full_df = pd.concat(all_dfs).sort_index()
    
    total_rows = len(full_df)
    print(f"Total pooled records: {total_rows}")
    
    # 2. Iterative Walk-Forward
    current_start = 0
    fold = 1
    results = []
    
    print(f"{'Fold':<5} | {'Start Date':<12} | {'End Date':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Signals'}")
    print("-" * 75)
    
    while current_start + train_size_rows + test_size_rows < total_rows:
        # Define Windows
        train_df = full_df.iloc[current_start : current_start + train_size_rows]
        test_df = full_df.iloc[current_start + train_size_rows : current_start + train_size_rows + test_size_rows]
        
        # Dates for reporting
        start_date = str(test_df.index[0].date())
        end_date = str(test_df.index[-1].date())
        
        # Setup Model
        model = StockModel()
        X_train = train_df[model.FEATURES]
        y_train = train_df[model.TARGET]
        X_test = test_df[model.FEATURES]
        y_test = test_df[model.TARGET]
        
        # Train
        model.train(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        signals = np.sum(y_pred)
        
        print(f"{fold:<5} | {start_date:<12} | {end_date:<12} | {acc:<10.4f} | {prec:<10.4f} | {signals}")
        
        results.append({
            'fold': fold,
            'accuracy': acc,
            'precision': prec,
            'signals': signals
        })
        
        # Advance Window
        current_start += test_size_rows
        fold += 1

    # 3. Final Summary
    if results:
        res_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print(" WALK-FORWARD SUMMARY ")
        print("="*50)
        print(f"Average Accuracy:  {res_df['accuracy'].mean():.4f}")
        print(f"Average Precision: {res_df['precision'].mean():.4f}")
        print(f"Total Signals:     {res_df['signals'].sum()}")
        print(f"Stability (Std Dev Precision): {res_df['precision'].std():.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Approximate settings: 2000 rows is roughly 8 years of data for one symbol, 
    # but since we POOL 13 symbols, 2000 rows is actually a much shorter temporal window.
    # We want a significant temporal window. 
    # Let's try 5000 rows for training (~1.5 years across all symbols) and 1000 for testing.
    walk_forward_validation(train_size_rows=5000, test_size_rows=1000)
