import os
import sys
import yaml

# Add the root directory to the path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.models.stock_model import StockModel
from services.models.model_trainer import StockModelTrainer

def load_config():
    """Load configuration from settings.yaml."""
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def run_training():
    """Main function to run the training pipeline with pooled data."""
    config = load_config()
    symbols = config.get("symbols", ["AAPL"])
    
    features_dir = "data/features"
    models_dir = "models"
    
    # Collect all feature files
    feature_files = [os.path.join(features_dir, f"{s}.parquet") for s in symbols]
    
    # Initialize the IA model
    model = StockModel()
    
    # Initialize the trainer
    trainer = StockModelTrainer(model)
    
    print(f"Starting GLOBAL training pipeline for symbols: {symbols}")
    
    try:
        # 1. Train on pooled data
        trainer.train_global_model(feature_files)
        
        # 2. Save the single global model
        trainer.save_model(models_dir, "global_stock_model")
        print("\nGlobal model successfully trained and saved.")
        
    except Exception as e:
        print(f"Error during global training: {e}")

if __name__ == "__main__":
    run_training()