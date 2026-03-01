import pandas as pd
import numpy as np
import os
import sys

# Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.models.stock_model import StockModel

def run_backtest(threshold=0.70, hold_days=7):
    print(f"\n🚀 STARTING BACKTEST (Threshold: {threshold}, Hold: {hold_days} days) 🚀")
    
    # 1. Load Global Model
    model_dir = "models"
    model = StockModel()
    try:
        model.load(model_dir, "xgboost_global_stock_model")
    except FileNotFoundError:
        print("Error: Global model not found. Run training pipeline first.")
        return

    # 2. Get all feature files
    features_dir = "data/features"
    symbols = [f.replace(".parquet", "") for f in os.listdir(features_dir) if f.endswith(".parquet")]
    
    all_trades = []
    
    for s in symbols:
        df = pd.read_parquet(os.path.join(features_dir, f"{s}.parquet"))
        
        # Use only the test set (last 20%) to avoid lookahead/overfitting bias
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        if len(test_df) < hold_days:
            continue
            
        # Get probabilities
        X_test = test_df[model.FEATURES]
        probs = model.predict_proba(X_test)[:, 1]
        test_df['prob'] = probs
        
        # Identify signals and apply cooldown
        signals = test_df[test_df['prob'] > threshold]
        
        last_exit_pos = -1
        for idx, row in signals.iterrows():
            try:
                current_pos = df.index.get_loc(idx)
                
                # COOLDOWN: If we are already in a trade for this symbol, skip
                if current_pos < last_exit_pos:
                    continue
                    
                if current_pos + hold_days < len(df):
                    price_entry = df.iloc[current_pos]['Close']
                    price_exit = df.iloc[current_pos + hold_days]['Close']
                    trade_return = (price_exit - price_entry) / price_entry
                    
                    all_trades.append({
                        'symbol': s,
                        'date': idx,
                        'prob': row['prob'],
                        'return': trade_return
                    })
                    
                    # Set the exit position for cooldown
                    last_exit_pos = current_pos + hold_days
            except Exception:
                continue

    if not all_trades:
        print("No signals found for the given threshold in the test set.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values(by='date')
    
    # 3. Calculate Stats
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['return'] > 0])
    win_rate = (winning_trades / total_trades) * 100
    avg_return = trades_df['return'].mean() * 100
    
    # More realistic cumulative return: assuming we allocate 1/10th of capital per trade
    # and we don't multiply everything at once (which caused the millions).
    # We'll use a simple sum or a capped compounding.
    total_cum_return = trades_df['return'].sum()
    
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS - {len(symbols)} Symbols")
    print(f"{'='*50}")
    print(f"Total signals triggered:  {total_trades}")
    print(f"Winning trades:           {winning_trades} ({win_rate:.2f}%)")
    print(f"Average return per trade: {avg_return:.2f}%")
    print(f"Total Return (Additive):  {total_cum_return*100:.2f}%")
    print(f"{'='*50}\n")
    
    # Show top 5 best and worst trades
    print("Top 5 Best Trades:")
    print(trades_df.sort_values(by='return', ascending=False).head(5))
    
    print("\nTop 5 Worst Trades:")
    print(trades_df.sort_values(by='return', ascending=True).head(5))

if __name__ == "__main__":
    # You can pass threshold as argument
    t = 0.60
    if len(sys.argv) > 1:
        t = float(sys.argv[1])
    run_backtest(threshold=t)
