import pandas as pd
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.models.stock_model import StockModel

def predict_latest(symbol, model):
    report = []
    header = f"\n{'='*50}\nPREDICTION FOR: {symbol}\n{'='*50}"
    print(header)
    report.append(header)
    
    # 1. Load latest features
    feature_file = f"data/features/{symbol}.parquet"
    if not os.path.exists(feature_file):
        msg = f"Error: No feature data found for {symbol}"
        print(msg)
        report.append(msg)
        return report
        
    df = pd.read_parquet(feature_file)
    
    # The last row should be the most recent data point
    latest_data = df.tail(1)
    latest_date = latest_data.index[0]
    
    # 2. Predict using the passed GLOBAL model
    prob = model.predict_proba(latest_data)[0][1]
    
    lines = [
        f"Latest Market Date: {latest_date}",
        f"Probability of >2% move (7d): {prob*100:.2f}%"
    ]
    
    # User requested threshold: 0.70 (70%) - High Conviction Mode
    if prob > 0.70:
        lines.append("SIGNAL: [BUY] (Strong High-Conviction Momentum)")
    else:
        lines.append("SIGNAL: [HOLD/WAIT]")
        
    lines.append(f"{'='*50}\n")
    
    for line in lines:
        print(line)
        report.append(line)
        
    return report
if __name__ == "__main__":
    model_dir = "models"
    
    # Load config to get target symbols
    import yaml
    config = {}
    try:
        with open("config/settings.yaml", "r") as f:
            config = yaml.safe_load(f)
    except:
        pass
    
    target_symbols = config.get("target_symbols", ["AAPL", "MSFT", "VRT", "AVGO", "SPY"])
    
    # Load the single global model once
    model = StockModel()
    model.load(model_dir, "xgboost_global_stock_model")
    
    # Predict ONLY for target symbols
    all_reports = []
    for symbol in target_symbols:
        report_lines = predict_latest(symbol, model)
        all_reports.extend(report_lines)
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    output_path = "data/latest_predictions.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))
    print(f"Results saved to {output_path}")
