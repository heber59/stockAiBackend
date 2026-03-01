# Stock AI Prediction System 🚀

This is the official backend for the Stock Prediction system based on Machine Learning and FastAPI. The project uses a quantitative pipeline to transform historical data into probabilistic predictions.

## ⚡ Master Command (Run Everything)
If you want to update data, train the global model, and see the latest signals in one go:
```bash
python3.12 pipelines/run_all.py
```

---

## 📌 Project Status

Currently, the system is capable of:
- **Data Download & Standardization:** Fetching historical prices from Yahoo Finance and Alpha Vantage with automatic handling of dividends and stock splits.
- **Advanced Feature Engineering:** Using `pandas-ta` to calculate 19 technical indicators (ATR, MACD, ADX, RSI, Log Returns, etc.).
- **Data Auditing:** Quality validation system and class balance check (Target: >2% price increase in 7 days).
- **AI Training:** Optimized XGBoost models using `Time-Series Split` to prevent look-ahead bias.
- **Persistence:** Saving trained models in JSON format with comprehensive training metadata.

## 🛠️ Pipeline Commands (WSL / Python 3.12)

Run these commands in order within your WSL terminal to operate the system:

### 1. Update Data
Download the latest market prices for the configured symbols.
```bash
python3.12 pipelines/pipeline_update_data.py
```

### 2. Generate Features
Transform raw data into numerical variables for the AI.
```bash
python3.12 pipelines/pipeline_generate_features.py
```

### 3. Audit & Quality Control
Verify that the data has no nulls and that the market signal is balanced.
```bash
python3.12 test/audit_features.py
```

### 4. Train the AI
Train the XGBoost model and generate the performance report (Accuracy/Precision).
```bash
python3.12 pipelines/pipeline_train_model.py
```

### 5. Run Live Inference (Test)
Get buy/hold signals and probabilities for the most recent market data.
```bash
python3.12 test/predict_recent.py
```

## 🔍 Development Utilities

- **Inspect Parquet:** Allows you to quickly visualize the content of data files from the terminal.
  ```bash
  python3.12 test/inspect_data.py data/features/AAPL.parquet
  ```

## 📂 Project Structure
- `services/market/`: Data fetching and standardization logic.
- `services/features/`: Technical indicator calculation engine (`FeatureEngineer`).
- `services/models/`: AI definition (`StockModel`) and training logic (`ModelTrainer`).
- `pipelines/`: Orchestration scripts for each stage of the process.
- `models/`: Trained model `.json` files.
- `data/`: Raw and processed (features) data in Parquet format.

---

## 🏗️ Infrastructure Summary (Original)

```text
Quant ML Production Pipeline 

LOCAL MACHINE
│
├── data pipeline
├── feature pipeline
├── training pipeline
└── model export

SUPABASE
│
├── predictions table
├── actual_results table
├── model_performance table

RENDER
│
├── loads trained model
└── serves predictions
```

*Developed for the Quant ML Production Pipeline.*