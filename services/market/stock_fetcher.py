import yfinance as yf
import pandas as pd
import requests
import time
from typing import Optional

class DataFetcherError(Exception):
    """Custom exception for DataFetcher errors."""
    pass

class DataFetcher:
    """Service to fetch stock data from multiple sources with fallback logic."""
    
    def __init__(self, alpha_vantage_api_key: Optional[str] = None):
        self.alpha_vantage_api_key = alpha_vantage_api_key

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Standardized interface to fetch data, trying yfinance first then Alpha Vantage."""
        data = self.fetch_from_yfinance(symbol)
        
        if data is None or data.empty:
            print(f"yfinance failed for {symbol}, trying Alpha Vantage...")
            data = self.fetch_from_alphavantage(symbol)
            
        if data is None or data.empty:
            raise DataFetcherError(f"Complete failure fetching data for symbol: {symbol}")
            
        return self._standardize_dataframe(data)

    def fetch_from_yfinance(self, symbol: str, period: str = "max") -> Optional[pd.DataFrame]:
        """Fetches data using yfinance library."""
        try:
            print(f"Attempting yfinance fetch for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"yfinance returned empty data for {symbol}.")
                return None
            return df
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}")
            return None

    def fetch_from_alphavantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches data from Alpha Vantage API."""
        if not self.alpha_vantage_api_key or self.alpha_vantage_api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
            print("Alpha Vantage fetch skipped: No valid API key provided.")
            return None
            
        try:
            print(f"Attempting Alpha Vantage fetch for {symbol}...")
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.alpha_vantage_api_key}'
            r = requests.get(url)
            data = r.json()
            
            if "Time Series (Daily)" not in data:
                print(f"Alpha Vantage error or no data for {symbol}: {data.get('Note', data.get('Error Message', 'Unknown error'))}")
                return None
                
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            # Standardize columns to match yfinance names
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            df = df.astype(float)
            return df
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
            return None

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the dataframe has a standardized format (DateTime index, specific columns)."""
        df.index = pd.to_datetime(df.index)
        # Ensure only common OHLCV columns exist and are named correctly
        cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        # In case of yfinance, keep Dividends/Stock Splits if they exist, or just filter.
        # For simplicity and consistency in ML, we focus on OHLCV.
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].copy()
        
        # Drop any rows with NaN in critical columns
        df.dropna(subset=["Close"], inplace=True)
        df.sort_index(inplace=True)
        
        # Deduplicate just in case
        df = df[~df.index.duplicated(keep='last')]
        
        return df
