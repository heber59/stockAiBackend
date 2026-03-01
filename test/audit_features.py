import pandas as pd
import numpy as np
import os

def audit_features(file_path):
    print(f"\n{'='*50}")
    print(f"AUDIT REPORT FOR: {file_path}")
    print(f"{'='*50}")
    
    df = pd.read_parquet(file_path)
    
    # 1. Basic Stats
    print(f"Total rows: {len(df)}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    
    # 2. Missing Values Check
    nulls = df.isnull().sum().sum()
    print(f"Total Null values: {nulls}")
    if nulls > 0:
        print("WARNING: Found null values in the following columns:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # 3. Individual Column Audits
    print("\n--- Feature Quality Checks ---")
    
    # RSI Check
    if 'RSI' in df.columns:
        within_range = df['RSI'].between(0, 100).all()
        print(f"RSI between 0-100: {'PASS' if within_range else 'FAIL'}")
        print(f"RSI Mean: {df['RSI'].mean():.2f}")
        
    # Trend/Volatility Checks (New features)
    for col in ['ATR', 'ADX', 'MACD']:
        if col in df.columns:
            print(f"{col} mean: {df[col].mean():.2f}")

    # Support/Resistance Check (Retained for 30 day window)
    if 'Support_30' in df.columns and 'Resistance_30' in df.columns:
        # Let's count how many times price is outside the 10/90 range
        outside_lower = (df['Close'] < df['Support_30']).sum()
        outside_upper = (df['Close'] > df['Resistance_30']).sum()
        print(f"Rows below Support_30: {outside_lower} ({outside_lower/len(df)*100:.1f}%)")
        print(f"Rows above Resistance_30: {outside_upper} ({outside_upper/len(df)*100:.1f}%)")

    # 4. Target Variable Audit (Multiclass: 0=Bearish, 1=Neutral, 2=Bullish)
    if 'Target' in df.columns:
        counts = df['Target'].value_counts(normalize=True).sort_index()
        print("\n--- Target Distribution (Class Balance) ---")
        class_labels = {0: 'BEARISH (<-2%)', 1: 'NEUTRAL', 2: 'BULLISH (>+2%)'}
        for cls, pct in counts.items():
            label = class_labels.get(int(cls), str(cls))
            print(f"Class {int(cls)} ({label}): {pct*100:.2f}%")
        
        if counts.get(2, 0) < 0.10:
            print("WARNING: BULLISH class is very rare (<10%). Model may struggle to detect upside.")
        if counts.get(0, 0) < 0.10:
            print("WARNING: BEARISH class is very rare (<10%). Model may struggle to detect downside.")

    # 5. VIX Regime Check
    if 'VIX_close' in df.columns:
        high_regime_pct = df['VIX_high_regime'].mean() * 100 if 'VIX_high_regime' in df.columns else 0
        print(f"\n--- VIX Regime ---")
        print(f"Rows in HIGH VOLATILITY regime (VIX>30): {high_regime_pct:.1f}%")

    # 6. Volatility Check
    if 'rolling_vol_30' in df.columns:
         print(f"Avg Volatility (rolling_vol_30): {df['rolling_vol_30'].mean():.4f}")

    print(f"{'='*50}\n")

if __name__ == "__main__":
    feat_dir = "data/features"
    if os.path.exists(feat_dir):
        files = [f for f in os.listdir(feat_dir) if f.endswith(".parquet")]
        for f in files[:2]: # Audit first 2 files
            audit_features(os.path.join(feat_dir, f))
    else:
        print("Features directory not found.")

