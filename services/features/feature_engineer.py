import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    """Service to transform raw market data into advanced technical features for ML models."""
    
    def __init__(self):
        pass

    def generate_features(self, df: pd.DataFrame, mkt_df: pd.DataFrame = None) -> pd.DataFrame:
        """Main method to calculate advanced features and the threshold-based target variable."""
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

        # 6. Market Context Features (SPY)
        if mkt_df is not None:
            mkt_df = mkt_df.copy()
            mkt_df['SPY_returns_1d'] = mkt_df['Close'].pct_change(1)
            mkt_df['SPY_returns_3d'] = mkt_df['Close'].pct_change(3)
            mkt_df['SPY_returns_7d'] = mkt_df['Close'].pct_change(7)
            mkt_df['SPY_volatility'] = mkt_df['SPY_returns_1d'].rolling(window=30).std()
            
            # Join with main df based on the index (Date)
            mkt_features = mkt_df[['SPY_returns_1d', 'SPY_returns_3d', 'SPY_returns_7d', 'SPY_volatility']]
            df = df.join(mkt_features, how='left')
        else:
            # Fill with 0 if no market data provided (for the SPY call itself or fallback)
            df['SPY_returns_1d'] = 0.0
            df['SPY_returns_3d'] = 0.0
            df['SPY_returns_7d'] = 0.0
            df['SPY_volatility'] = 0.0

        # 7. Target Variable (Threshold-based)
        future_close = df['Close'].shift(-7)
        df['Target'] = ((future_close - df['Close']) / df['Close'] > 0.02).astype(int)
        
        # 8. Numerical Stability: Handle Inf and NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows where critical features are NaN (e.g., from indicators), but KEEP rows with NaN targets
        feature_cols = [c for c in df.columns if c != 'Target']
        df.dropna(subset=feature_cols, inplace=True)
        
        return df
