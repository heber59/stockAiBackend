import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
from services.market.stock_fetcher import DataFetcher, DataFetcherError

def load_config():
    """Load configuration from settings.yaml."""
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def update_symbol_data(symbol, config, fetcher):
    """Orchestrates data fetching, merging, and saving for a single symbol."""
    data_dir = config.get("data_dir", "data/raw")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{symbol}.parquet")
    
    try:
        # 1. Fetch new data using the decoupled service
        new_data = fetcher.fetch_data(symbol)
        
        # 2. Merge with existing data if available
        if os.path.exists(file_path):
            print(f"Merging with existing data for {symbol}...")
            existing_data = pd.read_parquet(file_path)
            
            # Combine, deduplicate, and sort
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
        else:
            combined_data = new_data
            combined_data.sort_index(inplace=True)

        # 3. Save to Parquet
        print(f"Saving {len(combined_data)} rows to {file_path}")
        combined_data.to_parquet(file_path)
        
    except DataFetcherError as e:
        print(f"Pipeline error for {symbol}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {symbol}: {e}")

def main():
    config = load_config()
    symbols = config.get("symbols", [])
    av_key = config.get("alpha_vantage_api_key")
    
    # Initialize the decoupled service
    fetcher = DataFetcher(alpha_vantage_api_key=av_key)
    
    for symbol in symbols:
        update_symbol_data(symbol, config, fetcher)

if __name__ == "__main__":
    main()