import xgboost as xgb
import pandas as pd
import os
import json

class StockModel:
    """Encapsulates the XGBoost model and its feature requirements."""
    
    # Updated features list according to the new advanced engineering
    FEATURES = [
        'returns_1d', 'returns_3d', 'returns_5d', 'returns_7d', 'returns_10d', 'returns_14d',
        'ATR', 'rolling_vol_30', 
        'ADX', 'MACD', 'MACD_signal', 'MACD_hist',
        'RSI', 
        'volume_change', 'volume_sma_30', 'volume_ratio',
        'Support_30', 'Resistance_30', 'Dist_to_Support_30', 'Dist_to_Resistance_30',
        'SPY_returns_1d', 'SPY_returns_3d', 'SPY_returns_7d', 'SPY_volatility'
    ]
    
    TARGET = 'Target'
    
    def __init__(self, params=None):
        # Default robust parameters for financial binary classification
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the model using the provided features and target."""
        # Ensure only the required features are used and in the correct order
        X_filtered = X[self.FEATURES]
        self.model.fit(X_filtered, y)
        print("Model training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generates binary predictions (0 or 1)."""
        X_filtered = X[self.FEATURES]
        return self.model.predict(X_filtered)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates probability scores for each class."""
        X_filtered = X[self.FEATURES]
        return self.model.predict_proba(X_filtered)

    def save(self, model_dir: str, model_name: str = "xgboost_model"):
        """Saves the model and its metadata."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model in XGBoost JSON format
        model_path = os.path.join(model_dir, f"{model_name}.json")
        self.model.save_model(model_path)
        
        # Save metadata (features used, params, etc.)
        metadata = {
            "features": self.FEATURES,
            "target": self.TARGET,
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
