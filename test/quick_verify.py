import os
import sys
import shutil
import subprocess

def run_cmd(command):
    print(f"\n🏃 Executing: {command}")
    res = subprocess.run(command, shell=True, text=True)
    if res.returncode != 0:
        print(f"❌ Command failed: {command}")
        return False
    return True

def main():
    print("🌟 Starting End-to-End Stock AI Verification Script 🌟")
    
    config_path = "config/settings.yaml"
    backup_path = "config/settings.yaml.backup"
    
    # 1. Backup configuration
    if os.path.exists(config_path):
        print(f"📁 Backing up config to {backup_path}")
        shutil.copy(config_path, backup_path)
    else:
        print("❌ Error: config/settings.yaml not found.")
        sys.exit(1)
        
    try:
        # 2. Modify config to use a tiny subset of symbols
        print("⚙️ Modifying config to run on AAPL, MSFT, NVDA, and SPY...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        in_symbols = False
        for line in lines:
            if line.strip().startswith("symbols:"):
                in_symbols = True
                new_lines.append("symbols: [\n  \"AAPL\", \"MSFT\", \"NVDA\", \"SPY\"\n]\n")
                continue
            if in_symbols:
                if line.strip().endswith("]"):
                    in_symbols = False
                continue
            new_lines.append(line)
            
        with open(config_path, "w") as f:
            f.writelines(new_lines)
            
        print("✅ Config updated.")
        
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/features", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # 3. Fetch data for this subset of symbols
        print("\n📥 Fetching raw data (AAPL, MSFT, NVDA, SPY, VIX, Sectors)...")
        if not run_cmd("venv/bin/python pipelines/pipeline_update_data.py"):
            sys.exit(1)
            
        # 4. Generate features
        print("\n📊 Generating features for the subset...")
        if not run_cmd("venv/bin/python pipelines/pipeline_generate_features.py"):
            sys.exit(1)
            
        # 5. Train global models (both Production and Historical leakage-free models)
        print("\n🧠 Training Global Models (tuned hyperparams)...")
        if not run_cmd("venv/bin/python pipelines/pipeline_train_model.py"):
            sys.exit(1)
            
        # 6. Run backtest with threshold 0.60
        print("\n📈 Running Backtest (Leakage-free, Threshold: 0.60)...")
        if not run_cmd("venv/bin/python test/backtest_model.py 0.60"):
            sys.exit(1)
            
        # 7. Run Threshold Analysis
        print("\n🎯 Running Threshold Precision Analysis...")
        if not run_cmd("venv/bin/python test/analyze_thresholds.py"):
            sys.exit(1)
            
        print("\n🎉 End-to-End Verification Completed Successfully! 🎉")
        
    finally:
        # 8. Restore original configuration
        if os.path.exists(backup_path):
            print(f"\n🔄 Restoring config from {backup_path}")
            shutil.copy(backup_path, config_path)
            os.remove(backup_path)
            print("✅ Original config restored.")

if __name__ == "__main__":
    main()
