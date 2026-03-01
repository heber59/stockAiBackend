import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score
from services.models.stock_model import StockModel
from services.features.feature_engineer import FeatureEngineer

class StockModelTrainer:
    """
    Service to orchestrate the training of the StockModel.
    
    Key improvements:
    - Walk-Forward Validation: trains on expanding window, tests on next unseen period.
    - Outlier period exclusion: removes COVID crash from training data.
    - Multiclass evaluation: per-class precision/recall for Bearish/Neutral/Bullish.
    """
    
    def __init__(self, model: StockModel):
        self.stock_model = model
        self.fe = FeatureEngineer()

    def train_on_symbol(self, symbol: str, data_path: str):
        """Loads data for a symbol and trains the model using walk-forward validation."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Feature data not found at {data_path}")
            
        print(f"Loading feature data for {symbol} from {data_path}...")
        df = pd.read_parquet(data_path)
        self._walk_forward_and_train(df, symbol)

    def train_global_model(self, feature_files: list):
        """Pools data from all symbols and trains a single global model with walk-forward validation."""
        all_dfs = []

        print(f"Pooling data from {len(feature_files)} symbols for Global Model...")
        
        for file_path in feature_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
        
        full_df = pd.concat(all_dfs).sort_index()
        print(f"Total pooled data: {len(full_df)} rows.")
        self._walk_forward_and_train(full_df, "GLOBAL_MODEL")

    def _walk_forward_and_train(self, df: pd.DataFrame, label: str):
        """
        Walk-Forward Validation:
        - Splits data into N annual folds.
        - For each fold: trains on all prior data (excluding outlier periods), tests on current fold.
        - Final model is trained on ALL clean data for production use.
        
        This prevents look-ahead bias and gives a realistic estimate of live performance.
        """
        df = df.dropna(subset=[self.stock_model.TARGET])
        df = df.sort_index()
        
        # Determine fold boundaries (annual splits)
        years = sorted(df.index.year.unique())
        
        if len(years) < 3:
            # Not enough data for walk-forward — fall back to simple 80/20 split
            print(f"  [Walk-Forward] Not enough years for WF validation. Using 80/20 split.")
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            self._execute_training(train_df, test_df, label)
            return
        
        # Use last 20% of years as walk-forward test folds
        n_test_folds = max(1, len(years) // 5)
        test_years = years[-n_test_folds:]
        train_years = years[:-n_test_folds]
        
        print(f"\n  [Walk-Forward] Training years: {train_years[0]}-{train_years[-1]}")
        print(f"  [Walk-Forward] Test fold years: {test_years}\n")

        all_test_preds = []
        all_test_actuals = []
        
        for i, test_year in enumerate(test_years):
            # Train on all data up to (but not including) test_year
            train_cutoff = str(test_year - 1) + '-12-31'
            train_df = df[df.index <= train_cutoff]
            test_df = df[df.index.year == test_year]

            if len(train_df) < 100 or len(test_df) < 10:
                print(f"  [Walk-Forward] Fold {test_year}: insufficient data, skipping.")
                continue
            
            # Exclude outlier periods from training only
            train_df_clean = self.fe.filter_outlier_periods(train_df)
            
            X_train = train_df_clean[self.stock_model.FEATURES]
            y_train = train_df_clean[self.stock_model.TARGET].astype(int)
            X_test = test_df[self.stock_model.FEATURES]
            y_test = test_df[self.stock_model.TARGET].astype(int)
            
            self.stock_model.model.fit(X_train, y_train)
            y_pred = self.stock_model.model.predict(X_test)
            
            all_test_preds.extend(y_pred)
            all_test_actuals.extend(y_test)
            
            acc = accuracy_score(y_test, y_pred)
            print(f"  [Walk-Forward] Fold {test_year}: Accuracy={acc:.4f} (n_test={len(y_test)})")

        # Consolidated walk-forward report
        if all_test_preds:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD VALIDATION RESULTS — {label}")
            print(f"{'='*60}")
            print(classification_report(
                all_test_actuals, all_test_preds,
                target_names=['BEARISH', 'NEUTRAL', 'BULLISH'],
                zero_division=0
            ))

        # Final production model: train on ALL clean data
        print(f"\n  [Final Train] Training production model on all clean data for {label}...")
        full_clean = self.fe.filter_outlier_periods(df)
        X_all = full_clean[self.stock_model.FEATURES]
        y_all = full_clean[self.stock_model.TARGET].astype(int)
        self.stock_model.model.fit(X_all, y_all)
        print(f"  [Final Train] Production model ready. ({len(X_all)} rows)")

        # Feature importance
        importance = self.stock_model.model.feature_importances_
        feature_imp = pd.Series(importance, index=self.stock_model.FEATURES).sort_values(ascending=False)
        print(f"\nTop 10 Most Important Features ({label}):")
        print(feature_imp.head(10).to_string())

    def _execute_training(self, train_df, test_df, label):
        """Fallback simple train/test split (used when insufficient data for WF)."""
        train_df = train_df.dropna(subset=[self.stock_model.TARGET])
        test_df = test_df.dropna(subset=[self.stock_model.TARGET])

        X_train = train_df[self.stock_model.FEATURES]
        y_train = train_df[self.stock_model.TARGET].astype(int)
        X_test = test_df[self.stock_model.FEATURES]
        y_test = test_df[self.stock_model.TARGET].astype(int)

        self.stock_model.model.fit(X_train, y_train)
        
        y_pred = self.stock_model.model.predict(X_test)
        print(f"\n--- Metrics for {label} ---")
        print(classification_report(
            y_test, y_pred,
            target_names=['BEARISH', 'NEUTRAL', 'BULLISH'],
            zero_division=0
        ))

    def save_model(self, model_dir: str, symbol: str):
        """Saves the trained model to the specified directory."""
        self.stock_model.save(model_dir, f"xgboost_{symbol}")
