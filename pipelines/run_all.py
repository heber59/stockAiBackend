import subprocess
import sys
import os

def run_command(command, description, log_file=None):
    print(f"\n{'#'*60}")
    print(f"### STEP: {description}")
    print(f"{'#'*60}")
    
    # Use the same python interpreter
    cmd_list = [sys.executable] + command.split()
    
    try:
        # Capture output so we can save it and print it
        result = subprocess.run(cmd_list, check=True, text=True, capture_output=True)
        print(result.stdout)
        
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'#'*60}\n### STEP: {description}\n{'#'*60}\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"\nERRORS/WARNINGS:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {description}: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("🚀 STARTING FULL XGBOOST PIPELINE 🚀")
    
    os.makedirs("data", exist_ok=True)
    log_path = "data/pipeline_log.txt"
    # Clear previous log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- FULL PIPELINE EXECUTION LOG ---\n")

    steps = [
        ("pipelines/pipeline_update_data.py", "Updating Raw Market Data"),
        ("pipelines/pipeline_generate_features.py", "Generating Advanced Features"),
        ("test/audit_features.py", "Auditing Feature Quality"),
        ("pipelines/pipeline_train_model.py", "Training Global XGBoost Model"),
        ("test/backtest_model.py", "Backtesting PnL (Buy & Hold 7 days)"),
        ("test/predict_recent.py", "Generating Latest Predictions")
    ]
    
    for script, description in steps:
        if not run_command(script, description, log_path):
            print(f"\n⛔ Pipeline stopped due to errors. Check {log_path} for details.")
            sys.exit(1)
            
    print(f"\n✅ FULL PIPELINE EXECUTED SUCCESSFULLY ✅")
    print(f"Metrics and logs saved to: {log_path}")
    print("Final signals saved to: data/latest_predictions.txt")

if __name__ == "__main__":
    main()
