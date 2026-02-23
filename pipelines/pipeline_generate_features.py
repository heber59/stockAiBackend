import os
import sys
import pandas as pd

# Add the root directory to the path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.features.feature_engineer import FeatureEngineer

def process_all_features():
    """Orchestrates loading raw data, generating features, and saving results."""
    raw_dir = "data/raw"
    features_dir = "data/features"
    os.makedirs(features_dir, exist_ok=True)
    
    engineer = FeatureEngineer()
    
    # Check if raw data exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory not found at {raw_dir}")
        return
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    
    if not files:
        print("No .parquet files found in data/raw. Please run the data update pipeline first.")
        return
        
    print(f"Starting feature generation for {len(files)} symbols...")
    
    # 1. Load SPY data as the market context
    spy_path = os.path.join(raw_dir, "SPY.parquet")
    df_spy = None
    if os.path.exists(spy_path):
        print("Loading SPY as market context...")
        df_spy = pd.read_parquet(spy_path)
    else:
        print("Warning: SPY.parquet not found in data/raw. Features will be generated without market context.")

    for filename in files:
        symbol = filename.replace(".parquet", "")
        raw_path = os.path.join(raw_dir, filename)
        feature_path = os.path.join(features_dir, filename)
        
        try:
            print(f"Processing features for {symbol}...")
            # 2. Load raw data
            df_raw = pd.read_parquet(raw_path)
            
            # 3. Generate features (with SPY context)
            df_features = engineer.generate_features(df_raw, mkt_df=df_spy)
            
            # 4. Save to features directory
            print(f"Saving {len(df_features)} rows of engineered data to {feature_path}")
            df_features.to_parquet(feature_path)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    process_all_features()