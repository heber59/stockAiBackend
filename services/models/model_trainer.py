import pandas as pd
import os
from services.models.stock_model import StockModel

class StockModelTrainer:
    """Service to orchestrate the training of the StockModel."""
    
    def __init__(self, model: StockModel):
        self.stock_model = model

    def train_on_symbol(self, symbol: str, data_path: str):
        """Loads data for a symbol and trains the model using a time-series split."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Feature data not found at {data_path}")
            
        print(f"Loading feature data for {symbol} from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Chronological split: 80% training, 20% testing
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        self._execute_training(train_df, test_df, symbol)

    def train_global_model(self, feature_files: list):
        """Pools data from all symbols and trains a single global model."""
        all_train_dfs = []
        all_test_dfs = []

        print(f"Pooling data from {len(feature_files)} symbols for Global Model...")
        
        for file_path in feature_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue
                
            df = pd.read_parquet(file_path)
            
            # Split each symbol individually to maintain chronological order for each
            split_idx = int(len(df) * 0.8)
            all_train_dfs.append(df.iloc[:split_idx])
            all_test_dfs.append(df.iloc[split_idx:])
        
        # Concatenate everything
        full_train_df = pd.concat(all_train_dfs).sort_index()
        full_test_df = pd.concat(all_test_dfs).sort_index()
        
        print(f"Global Training Set: {len(full_train_df)} rows. Global Test Set: {len(full_test_df)} rows.")
        self._execute_training(full_train_df, full_test_df, "GLOBAL_MODEL")

    def _execute_training(self, train_df, test_df, label):
        """Internal method to run the actual fit and evaluation."""
        X_train = train_df[self.stock_model.FEATURES]
        y_train = train_df[self.stock_model.TARGET]
        
        X_test = test_df[self.stock_model.FEATURES]
        y_test = test_df[self.stock_model.TARGET]
        
        self.stock_model.train(X_train, y_train)
        self._evaluate(X_test, y_test, label)

    def _evaluate(self, X_test, y_test, symbol):
        """Calculates and prints performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, classification_report
        
        print(f"Evaluating model for {symbol}...")
        y_pred = self.stock_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print(f"\n--- Metrics for {symbol} ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision (Buy Signal): {precision:.4f}")
        print("\nFull Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature Importance
        import matplotlib.pyplot as plt
        # Note: In a headless or remote environment, we might just log this.
        # But for now, we can print the top features from the model
        importance = self.stock_model.model.feature_importances_
        feature_imp = pd.Series(importance, index=self.stock_model.FEATURES).sort_values(ascending=False)
        print("\nTop 5 Most Important Features:")
        print(feature_imp.head(5))

    def save_model(self, model_dir: str, symbol: str):
        """Saves the trained model to the specified directory."""
        self.stock_model.save(model_dir, f"xgboost_{symbol}")
