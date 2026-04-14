# Stock Prediction Project: XGBoost Explained

This document provides a detailed explanation of the stock prediction system built for this project. It is designed to be shared with a teacher or evaluator.

---

## 1. How XGBoost Works (The Simple Version)
**XGBoost** (Extreme Gradient Boosting) is a powerful "team-based" machine learning algorithm. Imagine you have a team of 150 students (trees) trying to predict if a stock will go up or down:

1.  **The First Student:** Makes a simple guess. It won't be perfect.
2.  **The Second Student:** Doesn't start from scratch. Instead, they look at where the first student made mistakes and tries to fix them.
3.  **The Subsequent Students:** Each one focuses on correcting the errors of the team before them.

By the end, the "weak" individual guesses are combined into a very strong and accurate final prediction.

---

## 2. Model Parameters
In the project code (`stock_model.py`), we configured XGBoost with these specific settings:

| Parameter | Value | What it does |
| :--- | :--- | :--- |
| `n_estimators` | 150 | The number of trees (students) in the team. More trees can learn more, but too many can cause overfitting. |
| `max_depth` | 5 | How "tall" each tree can grow. A depth of 5 allows the model to find complex patterns without getting lost in noise. |
| `learning_rate` | 0.08 | How fast the model learns. We use a small number so the model "drifts" toward the correct answer carefully. |
| `subsample` | 0.8 | Each tree only sees 80% of the data. This keeps the trees different from each other (diversity). |
| `colsample_bytree` | 0.8 | Each tree only sees 80% of the features (indicators). This prevents the model from relying too much on one single indicator like RSI. |
| `objective` | `multi:softprob`| Tells XGBoost we have 3 categories (Bearish, Neutral, Bullish) and we want the probability for each. |

---

## 3. Real Data Showcase (AAPL)
The model analyzes historical data for stocks like **AAPL**. Here is a sample of the actual data used for training (after processing):

| Date | Close Price | returns_1d | RSI | ATR | Target (0=Bearish, 1=Neutral, 2=Bullish) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-02-18 | 264.35 | -1.61% | 51.63 | 6.55 | 0 |
| 2026-02-19 | 260.57 | -1.42% | 45.45 | 6.50 | 0 |
| 2026-02-20 | 264.57 | +1.53% | 49.70 | 6.50 | 0 |
| 2026-02-23 | 266.17 | +0.60% | 51.33 | 6.47 | 0 |
| 2026-02-24 | 272.14 | +2.23% | 56.94 | 6.63 | 0 |

### Column Explanations (The Features)
We transformed raw prices into "Features" that the model can understand:

1.  **Returns (1d, 3d, 7d, etc.):** The percentage change in price. This tells the model if the stock is currently in a momentum phase.
2.  **RSI (Relative Strength Index):** Measures if a stock is "Overbought" (too expensive) or "Oversold" (too cheap).
3.  **MACD:** A trend-following indicator. When the MACD line crosses the signal line, it often signals a change in direction.
4.  **ATR (Average True Range):** Measures **Volatility**. It tells the model how "jumpy" the stock is.
5.  **Volume Ratio:** Compares today's trading volume to the 30-day average. High volume often confirms a price move.
6.  **Support & Resistance:** Key price levels where the stock historically stops falling (Support) or stops rising (Resistance).
7.  **Market Features (SPY & VIX):** The model also looks at the overall S&P 500 (SPY) and the "Fear Index" (VIX) to understand the global economic mood.

---

## 4. Prediction Strategy
The model doesn't just guess a price; it classifies the next 7 days into three buckets:
- **Class 0 (Bearish):** Expecting a drop of more than 2%.
- **Class 1 (Neutral):** Expecting the price to stay mostly flat (+/- 2%).
- **Class 2 (Bullish):** Expecting a gain of more than 2%.

We use **Walk-Forward Validation**, which means the model is always tested on data it has never seen before (simulating "tomorrow" over and over again) to ensure it works in the real world.
