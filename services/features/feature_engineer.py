import pandas as pd
import numpy as np
import pandas_ta as ta

def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Strips timezone info from a DatetimeIndex and normalizes to date-only.
    Handles both tz-aware (tz_convert) and tz-naive (normalize only) indexes.
    """
    idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()

class FeatureEngineer:
    """Service to transform raw market data into advanced technical features for ML models."""
    
    # Date ranges to exclude from training due to extreme outlier behavior
    # (e.g., COVID crash). This prevents the model from overfitting to rare events.
    EXCLUDE_RANGES = [
        ('2020-02-01', '2020-05-31'),  # COVID crash / V-recovery
    ]
    
    def __init__(self):
        pass


    def generate_features(self, df: pd.DataFrame, mkt_df: pd.DataFrame = None, vix_df: pd.DataFrame = None, symbol: str = None, data_dir: str = "data/raw", config: dict = None) -> pd.DataFrame:
        """
        Main method to calculate advanced features and the ALPHA multiclass target.
        
        Target classes (Alpha = Stock Return - SPY Return):
            0 = Bearish Alpha (Alpha < -1.5%)
            1 = Neutral Alpha (Alpha between -1.5% and 1.5%)
            2 = Bullish Alpha (Alpha > 1.5%)
        """

        df = df.copy()
        df.sort_index(inplace=True)
        
        # Ensure we have clean float data for ta
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        # 1. Returns (1d, 3d, 5d, 7d, 10d, 14d)
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_7d'] = df['Close'].pct_change(7)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_14d'] = df['Close'].pct_change(14)
        
        # 2. Volatility Features
        df['ATR'] = df.ta.atr(length=14)
        df['rolling_vol_30'] = df['returns_1d'].rolling(window=30).std()
        
        # 3. Trend Indicators
        adx_df = df.ta.adx(length=14)
        if adx_df is not None:
            df['ADX'] = adx_df['ADX_14']
        
        macd_df = df.ta.macd(fast=12, slow=26, signal=9)
        if macd_df is not None:
            df['MACD'] = macd_df['MACD_12_26_9']
            df['MACD_signal'] = macd_df['MACDs_12_26_9']
            df['MACD_hist'] = macd_df['MACDh_12_26_9']
            
        df['RSI'] = df.ta.rsi(length=14)
        
        # 3.1 Directional / Trend Strength Features (To reduce volatility dominance)
        # SMA Slopes (Captures trend direction)
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_20_slope'] = df['sma_20'].pct_change(5) # 5-day slope
        df['sma_50_slope'] = df['sma_50'].pct_change(5)
        
        # EMA Crossovers
        df['ema_12'] = df.ta.ema(length=12)
        df['ema_26'] = df.ta.ema(length=26)
        df['ema_dist'] = (df['ema_12'] - df['ema_26']) / df['ema_26']

        # Price Location Relative to Range (Relative Strength)
        df['high_52w'] = df['High'].rolling(window=252, min_periods=50).max()
        df['low_52w'] = df['Low'].rolling(window=252, min_periods=50).min()
        df['dist_from_high_52w'] = (df['high_52w'] - df['Close']) / df['high_52w']
        df['dist_from_low_52w'] = (df['Close'] - df['low_52w']) / df['low_52w']
        
        # 4. Volume Features
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_sma_30'] = df['Volume'].rolling(window=30).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_30']
        
        # 5. Price Levels
        for window in [30]:
            df[f'Support_{window}'] = df['Low'].rolling(window=window).apply(lambda x: np.percentile(x, 10))
            df[f'Resistance_{window}'] = df['High'].rolling(window=window).apply(lambda x: np.percentile(x, 90))
            df[f'Dist_to_Support_{window}'] = (df['Close'] - df[f'Support_{window}']) / df[f'Support_{window}']
            df[f'Dist_to_Resistance_{window}'] = (df[f'Resistance_{window}'] - df['Close']) / df[f'Resistance_{window}']

        # Normalize df index to tz-naive BEFORE any join operations
        # This prevents "Cannot join tz-naive with tz-aware DatetimeIndex" errors
        df.index = _normalize_index(df.index)

        # 6. Market Context Features (SPY)
        if mkt_df is not None:
            mkt_df = mkt_df.copy()
            mkt_df.index = _normalize_index(mkt_df.index)
            mkt_df['SPY_returns_1d'] = mkt_df['Close'].pct_change(1)
            mkt_df['SPY_returns_3d'] = mkt_df['Close'].pct_change(3)
            mkt_df['SPY_returns_7d'] = mkt_df['Close'].pct_change(7)
            mkt_df['SPY_volatility'] = mkt_df['SPY_returns_1d'].rolling(window=30).std()
            
            mkt_features = mkt_df[['SPY_returns_1d', 'SPY_returns_3d', 'SPY_returns_7d', 'SPY_volatility']]
            df = df.join(mkt_features, how='left')
        else:
            df['SPY_returns_1d'] = 0.0
            df['SPY_returns_3d'] = 0.0
            df['SPY_returns_7d'] = 0.0
            df['SPY_volatility'] = 0.0

        # 7. Market Regime Filter (VIX-based)
        if vix_df is not None:
            vix_df = vix_df.copy()
            vix_df.index = _normalize_index(vix_df.index)
            df['VIX_close'] = vix_df['Close'].reindex(df.index, method='ffill')
            df['VIX_high_regime'] = (df['VIX_close'] > 30).astype(int)
        else:
            df['VIX_close'] = 0.0
            df['VIX_high_regime'] = 0

        # --- WEEKLY ROADMAP ADDITIONS ---

        # 7.1 WEEK 2: Earnings Context
        df['days_to_earnings'] = 99.0 
        try:
            earn_file = os.path.join(data_dir, "earnings_calendar.csv")
            if os.path.exists(earn_file) and symbol:
                earn_df = pd.read_csv(earn_file, index_col=0)
                if symbol in earn_df.index:
                    next_earn = pd.to_datetime(earn_df.loc[symbol, 'next_earnings_date']).tz_localize(None)
                    diff = (next_earn - df.index).days
                    df['days_to_earnings'] = diff.clip(lower=-30, upper=99)
        except:
            pass

        # 7.2 WEEK 3: Sector Relative Strength
        df['rel_strength_sector'] = 0.0
        if config and symbol:
            s_map = config.get("sector_map", {})
            sector_ticker = s_map.get(symbol)
            if sector_ticker:
                sector_file = os.path.join(data_dir, f"{sector_ticker}.parquet")
                if os.path.exists(sector_file):
                    sec_df = pd.read_parquet(sector_file)
                    sec_df.index = _normalize_index(sec_df.index)
                    sec_returns = sec_df['Close'].pct_change(5) 
                    stock_returns = df['Close'].pct_change(5)
                    df['rel_strength_sector'] = stock_returns - sec_returns.reindex(df.index, method='ffill')

        # 8. ALPHA Multiclass Target Variable (Relative Return vs SPY)
        if mkt_df is not None:
            future_stock_close = df['Close'].shift(-7)
            stock_return_7d = (future_stock_close - df['Close']) / df['Close']
            
            mkt_df['SPY_future_close'] = mkt_df['Close'].shift(-7)
            spy_return_7d = (mkt_df['SPY_future_close'] - mkt_df['Close']) / mkt_df['Close']
            spy_return_aligned = spy_return_7d.reindex(df.index, method='ffill')
            
            # Alpha = Stock Return - Index Return
            alpha_7d = stock_return_7d - spy_return_aligned
            
            df['Target'] = 1 # Neutral alpha
            df.loc[alpha_7d < -0.015, 'Target'] = 0 # Significant underperformance
            df.loc[alpha_7d > 0.015, 'Target'] = 2  # Significant outperformance
            df['Target'] = df['Target'].where(alpha_7d.notna(), other=np.nan)
        else:
            future_close = df['Close'].shift(-7)
            future_return = (future_close - df['Close']) / df['Close']
            df['Target'] = 1
            df.loc[future_return < -0.02, 'Target'] = 0
            df.loc[future_return > 0.02, 'Target'] = 2
            df['Target'] = df['Target'].where(future_return.notna(), other=np.nan)

        # 9. Numerical Stability: Handle Inf and NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_cols = [c for c in df.columns if c != 'Target']
        df.dropna(subset=feature_cols, inplace=True)
        df.dropna(subset=['Target'], inplace=True)
        df['Target'] = df['Target'].astype(int)
        
        return df


    def filter_outlier_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes known outlier date ranges from training data (e.g., COVID crash).
        This prevents the model from overfitting extreme, non-recurring events.
        """
        mask = pd.Series(True, index=df.index)
        for start, end in self.EXCLUDE_RANGES:
            in_range = (df.index >= start) & (df.index <= end)
            mask = mask & ~in_range
        removed = (~mask).sum()
        if removed > 0:
            print(f"  [Regime Filter] Removed {removed} rows from outlier periods (e.g., COVID crash).")
        return df[mask]
