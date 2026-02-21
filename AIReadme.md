# Stock AI Prediction System

## Overview

Stock AI Prediction System is a quantitative machine learning platform designed to analyze stock market data and generate probabilistic predictions of future price movements using advanced feature engineering and gradient boosting models.

The system uses historical financial data, statistical analysis, and machine learning to estimate the probability of a stock increasing or decreasing in value over a defined time horizon (initially 7 days).

The platform follows a production-grade architecture separating data ingestion, feature engineering, model training, and prediction serving.

---

## Core Objective

The primary goal of this system is to build a continuously improving AI model capable of:

- Analyzing historical stock market data
- Identifying statistical and temporal patterns
- Generating probabilistic predictions of future price movements
- Continuously improving accuracy through retraining with new data
- Serving predictions via a FastAPI backend to a Next.js frontend

The system prioritizes probability-based outputs rather than deterministic predictions, as financial markets are inherently stochastic.

---

## System Architecture

The system consists of three main components:

### 1. Local Training Environment

Runs on the developer’s local machine and is responsible for:

- Downloading historical stock data
- Maintaining and updating datasets
- Performing feature engineering
- Training machine learning models
- Evaluating model performance
- Exporting trained models for production use

---

### 2. Backend Prediction Service

Runs on a cloud server using FastAPI and is responsible for:

- Loading trained models
- Fetching latest market data
- Generating features in real time
- Producing predictions
- Returning analysis results via REST API
- Logging predictions to database

This service does **not train models**. It only performs inference.

---

### 3. Database Layer

PostgreSQL database hosted on Supabase stores:

- Prediction history
- Model performance metrics
- Actual market results for evaluation
- Training metadata

This enables continuous improvement and performance tracking.

---

## Machine Learning Model Strategy

The system uses a probabilistic machine learning approach.

### Primary model

- XGBoost (Extreme Gradient Boosting Classifier)

### Future ensemble model

- XGBoost (primary)
- LSTM Neural Network (secondary)

### Weighted voting system

```python
final_probability =
0.7 * XGBoost +
0.3 * LSTM

This ensemble approach improves prediction robustness and accuracy.

---

## Feature Engineering

The model does not use raw price data directly. Instead, it uses engineered statistical features including:

- Daily percent change
- Moving averages (7-day, 30-day)
- Average volume
- Volatility (standard deviation)
- Momentum indicators
- Statistical mean price
- Support level (10th percentile)
- Resistance level (90th percentile)
- Distance to support and resistance
- Trend indicators

These features allow the model to detect complex market patterns.

---

## Continuous Training Strategy

The system follows a walk-forward continuous training approach.

### Workflow

1. Historical data is collected and stored locally
2. Features are generated from raw data
3. Model is trained on full dataset
4. Model is validated on recent data
5. Model is exported for production use
6. New market data is periodically added
7. Model is retrained using expanded dataset

This ensures the model continuously improves over time.

The model never discards historical data, preventing catastrophic forgetting.

---

## Data Pipeline Structure

### Project directory structure

```text
stock-ai-system/

data/
  raw/
  features/

models/
  xgboost_model.pkl
  metadata.json

pipelines/
  pipeline_update_data.py
  pipeline_generate_features.py
  pipeline_train_model.py
  pipeline_retrain.py

services/
  market/
    stock_fetcher.py

config/
  settings.yaml


## Pipeline Responsibilities

### pipeline_update_data.py
Orchestrates the update process. It calls the `DataFetcher` service, merges new market data with historical records, and ensures persistent storage in optimized Parquet format.

### services/market/stock_fetcher.py
A decoupled service responsible only for data acquisition. It handles communication with `yfinance` and `Alpha Vantage`, implements retry/fallback logic, and standardizes raw data into a consistent internal format.

### pipeline_generate_features.py
Transforms raw data into machine learning features.

### pipeline_train_model.py
Trains the XGBoost model and evaluates performance.

### pipeline_retrain.py
Executes the full training pipeline end-to-end.

---

## Prediction Output

The system produces structured probabilistic analysis including:

- Current price
- Previous price
- Percent change
- Moving averages
- Support and resistance levels
- Trend classification
- Probability of upward movement
- Probability of downward movement
- Predicted price estimate
- Model confidence score
- Technical analysis summary

---

## Deployment Architecture

### Training environment
Local machine with CPU/GPU acceleration

### Backend
FastAPI hosted on Render

### Frontend
Next.js hosted on Vercel

### Database
Supabase PostgreSQL

---

## Design Principles

This system follows key machine learning engineering principles:

- Separation of training and inference
- Deterministic feature generation
- Reproducible training pipelines
- Continuous improvement through retraining
- Probabilistic predictions instead of deterministic outputs
- Offline-first dataset management
- Scalable modular architecture

---

## Long-Term Goals

- Improve prediction accuracy through continuous training
- Implement ensemble models combining XGBoost and LSTM
- Expand feature engineering with advanced indicators
- Support multiple stocks simultaneously
- Track long-term model performance
- Optimize training using GPU acceleration