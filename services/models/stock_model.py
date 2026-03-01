import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json

class StockModel:
    """
    Encapsulates the XGBoost model for directional multiclass stock prediction.
    
    Target Classes:
        0 = Bearish  (future 7d return < -2%)
        1 = Neutral  (-2% <= future 7d return <= 2%)
        2 = Bullish  (future 7d return > +2%)
    """
    
    FEATURES = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_7d', 'returns_10d', 'returns_14d',
        'ATR', 'rolling_vol_30', 
        'ADX', 'MACD', 'MACD_signal', 'MACD_hist',
        'RSI', 
        'volume_change', 'volume_sma_30', 'volume_ratio',
        'Support_30', 'Resistance_30', 'Dist_to_Support_30', 'Dist_to_Resistance_30',
        'SPY_returns_1d', 'SPY_returns_3d', 'SPY_returns_7d', 'SPY_volatility',
        'VIX_close', 'VIX_high_regime',
    ]
    
    TARGET = 'Target'
    CLASS_LABELS = {0: 'BEARISH', 1: 'NEUTRAL', 2: 'BULLISH'}
    NUM_CLASSES = 3
    
    def __init__(self, params=None):
        # Robust parameters for 3-class financial classification
        self.params = params or {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.08,
            'objective': 'multi:softprob',
            'num_class': self.NUM_CLASSES,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the model using the provided features and target."""
        X_filtered = X[self.FEATURES]
        self.model.fit(X_filtered, y)
        print("Model training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates class predictions (0=Bearish, 1=Neutral, 2=Bullish)."""
        X_filtered = X[self.FEATURES]
        return self.model.predict(X_filtered)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates class probability scores.
        Returns array of shape (n_samples, 3):
            col 0 = P(Bearish), col 1 = P(Neutral), col 2 = P(Bullish)
        """
        X_filtered = X[self.FEATURES]
        return self.model.predict_proba(X_filtered)

    def predict_directional(self, X: pd.DataFrame, bullish_threshold: float = 0.52, bearish_threshold: float = 0.52):
        """
        Generates a directional signal with confidence thresholds.
        Returns a dict with signal and probability breakdown.
        """
        probas = self.predict_proba(X)
        latest = probas[-1]  # most recent row
        p_bear, p_neutral, p_bull = latest
        
        # Determine signal based on thresholds
        if p_bull >= bullish_threshold:
            signal = 'BULLISH'
        elif p_bear >= bearish_threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'p_bullish': round(float(p_bull) * 100, 2),
            'p_bearish': round(float(p_bear) * 100, 2),
            'p_neutral': round(float(p_neutral) * 100, 2),
        }

    def save(self, model_dir: str, model_name: str = "xgboost_model"):
        """Saves the model and its metadata."""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_name}.json")
        self.model.save_model(model_path)
        
        metadata = {
            "features": self.FEATURES,
            "target": self.TARGET,
            "num_classes": self.NUM_CLASSES,
            "class_labels": self.CLASS_LABELS,
            "params": self.params
        }
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model and metadata saved to {model_dir}")

    def load(self, model_dir: str, model_name: str = "xgboost_model"):
        """Loads the model from disk."""
        model_path = os.path.join(model_dir, f"{model_name}.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
            
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")
