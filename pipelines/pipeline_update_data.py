import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
import yfinance as yf
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
        new_data = fetcher.fetch_data(symbol)
        
        if os.path.exists(file_path):
            print(f"Merging with existing data for {symbol}...")
            existing_data = pd.read_parquet(file_path)
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
        else:
            combined_data = new_data
            combined_data.sort_index(inplace=True)

        print(f"Saving {len(combined_data)} rows to {file_path}")
        combined_data.to_parquet(file_path)
        
    except DataFetcherError as e:
        print(f"Pipeline error for {symbol}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {symbol}: {e}")

def update_vix_data(data_dir):
    """
    Downloads VIX index data directly via yfinance.
    """
    file_path = os.path.join(data_dir, "^VIX.parquet")
    print("Fetching VIX data (market regime filter)...")
    try:
        vix = yf.download("^VIX", period="max", auto_adjust=True, progress=False)
        if vix.empty: return
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        vix.index = pd.to_datetime(vix.index).normalize()
        vix.sort_index(inplace=True)
        vix.to_parquet(file_path)
        print(f"VIX data saved.")
    except Exception as e:
        print(f"Error fetching VIX data: {e}")

def update_sector_data(data_dir):
    """
    Downloads Sector ETFs for relative strength analysis.
    XLK (Tech), XLF (Finance), XLE (Energy), XLV (Health), XLY (Retail), XLU (Utilities), XLI (Ind).
    """
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLU", "XLI", "XLB", "XLC"]
    print(f"Fetching Sector ETFs: {sectors}")
    try:
        data = yf.download(sectors, period="max", auto_adjust=True, progress=False)
        if data.empty: return
        
        # Save each sector ETF as its own parquet for easy joining
        for ticker in sectors:
            ticker_data = data.xs(ticker, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
            if ticker_data.empty: continue
            ticker_data.index = pd.to_datetime(ticker_data.index).normalize()
            ticker_data.to_parquet(os.path.join(data_dir, f"{ticker}.parquet"))
        print("Sector data saved.")
    except Exception as e:
        print(f"Error fetching sector data: {e}")

def update_earnings_dates(symbols, data_dir):
    """
    Fetches next earnings dates for all symbols and saves to a central CSV.
    """
    print("Fetching Earnings Calendar dates...")
    earnings_map = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # Typically 'Earnings Date' is the first key
                edate = cal.iloc[0,0] # Extract the date
                earnings_map[symbol] = edate
        except:
            continue
    
    if earnings_map:
        df = pd.DataFrame.from_dict(earnings_map, orient='index', columns=['next_earnings_date'])
        df.to_csv(os.path.join(data_dir, "earnings_calendar.csv"))
        print(f"Earnings calendar saved for {len(df)} symbols.")


def main():
    config = load_config()
    symbols = config.get("symbols", [])
    av_key = config.get("alpha_vantage_api_key")
    data_dir = config.get("data_dir", "data/raw")
    
    # Initialize the decoupled service
    fetcher = DataFetcher(alpha_vantage_api_key=av_key)
    
    for symbol in symbols:
        update_symbol_data(symbol, config, fetcher)

    # Weekly Roadmap Enhancements
    update_vix_data(data_dir)
    update_sector_data(data_dir)
    update_earnings_dates(symbols, data_dir)


if __name__ == "__main__":
    main()