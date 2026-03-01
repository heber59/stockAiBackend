import os
import sys
import pandas as pd
import yaml

# Add the root directory to the path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.features.feature_engineer import FeatureEngineer

def load_config():
    """Load configuration from settings.yaml."""
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except:
        return {}

def load_vix_data(raw_dir):
    """Loads VIX data from disk if it was downloaded, otherwise returns None."""
    vix_path = os.path.join(raw_dir, "^VIX.parquet")
    if os.path.exists(vix_path):
        print("Loading VIX as market regime context...")
        return pd.read_parquet(vix_path)
    # Fallback: try UVXY as VIX proxy
    uvxy_path = os.path.join(raw_dir, "UVXY.parquet")
    if os.path.exists(uvxy_path):
        print("Loading UVXY as VIX proxy for market regime context...")
        return pd.read_parquet(uvxy_path)
    print("Warning: No VIX data found. Regime filter will be disabled.")
    return None

def process_all_features():
    """Orchestrates loading raw data, generating features, and saving results."""
    raw_dir = "data/raw"
    features_dir = "data/features"
    os.makedirs(features_dir, exist_ok=True)
    
    engineer = FeatureEngineer()
    
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory not found at {raw_dir}")
        return
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    
    if not files:
        print("No .parquet files found in data/raw. Please run the data update pipeline first.")
        return
        
    print(f"Starting feature generation for {len(files)} symbols...")
    
    # Load SPY as market context
    spy_path = os.path.join(raw_dir, "SPY.parquet")
    df_spy = None
    if os.path.exists(spy_path):
        print("Loading SPY as market context...")
        df_spy = pd.read_parquet(spy_path)
    else:
        print("Warning: SPY.parquet not found. Features will be generated without market context.")

    # Load VIX as regime filter
    df_vix = load_vix_data(raw_dir)
    config = load_config()

    for filename in files:
        symbol = filename.replace(".parquet", "")
        # Skip sector ETFs
        if symbol in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLU", "XLI", "XLB", "XLC", "GLD", "SLV", "TLT", "HYG", "^VIX"]:
            continue

        raw_path = os.path.join(raw_dir, filename)
        feature_path = os.path.join(features_dir, filename)
        
        try:
            print(f"Processing features for {symbol}...")
            df_raw = pd.read_parquet(raw_path)
            
            # Generate features with full Week 2-4 context
            df_features = engineer.generate_features(
                df_raw, 
                mkt_df=df_spy, 
                vix_df=df_vix, 
                symbol=symbol, 
                data_dir=raw_dir, 
                config=config
            )
            
            print(f"Saving {len(df_features)} rows of engineered data to {feature_path}")
            df_features.to_parquet(feature_path)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    process_all_features()