# LSTM Price Prediction — Financial Time Series Research Baseline

This project implements a complete end-to-end LSTM-based financial forecasting pipeline using Bitcoin historical market data.

The goal of the project is not merely to “predict prices,” but to build a statistically correct and extensible quantitative research framework for experimenting with:

- financial time-series preprocessing
- feature engineering
- sequential neural networks
- forecasting methodologies
- evaluation and visualization
- trading-oriented ML experimentation

---

# Project Overview

The pipeline includes:

- Historical crypto market data ingestion
- Log return computation
- Feature engineering
- Sequence generation for LSTM input
- Multivariate LSTM modeling
- Regression and classification experiments
- Train/test temporal splitting
- Leakage prevention
- Prediction visualization
- Directional accuracy evaluation

---

# Project Structure

```text
lstm-price-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── lstm.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── backtest.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   └── logger.py
│   │
│   └── main.py
│
├── configs/
│   └── config.yaml
│
├── experiments/
│   └── logs/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- yFinance
- Matplotlib
- Scikit-learn

---

# Features Engineered

The project currently uses:

| Feature | Description |
|---|---|
| Returns | Log returns |
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| Rolling Volatility | Rolling standard deviation |
| RSI | Relative Strength Index |

---

# Modeling Approaches

## 1. Return Regression

Predicting future log returns directly using:

\[
r_{t+1}
\]

Result:
- low MSE
- predictions collapsed toward near-zero values
- weak directional signal

---

## 2. Direction Classification

Predicting:

\[
P(r_{t+1} > 0)
\]

using binary classification.

Result:
- classification accuracy near random (~50%)
- probabilities converged near 0.47–0.50
- weak predictive separability

---

# Important Findings

This project demonstrates several important financial ML realities:

- Next-step financial returns are extremely noisy
- Low prediction loss does not imply profitable forecasting
- Financial models often collapse toward mean predictions
- Classical technical indicators alone provide weak signal
- Proper train/test splitting and leakage prevention are critical

---

# Visualization

The project includes prediction visualization for:

- actual vs predicted returns
- classification probabilities
- model behavior inspection

This helped identify:
- volatility suppression
- mean-collapse behavior
- weak directional confidence

---

# Example Workflow

```text
Market Data
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Scaling
    ↓
Sequence Generation
    ↓
LSTM Modeling
    ↓
Evaluation
    ↓
Visualization
```

---

# Running the Project

## Create virtual environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

---

# Install dependencies

```powershell
pip install -r requirements.txt
```

---

# Run the project

```powershell
python -m src.main
```

---

# Key Lessons Learned

This project evolved from a simple LSTM tutorial into a genuine quantitative research baseline.

Major lessons include:

- leakage prevention in time-series data
- multivariate sequential modeling
- statistical interpretation of ML outputs
- difference between prediction and decision systems
- limitations of naive financial forecasting

---

# Future Improvements

Potential future extensions:

- Multi-horizon forecasting
- Volatility prediction
- Transformer architectures
- Attention mechanisms
- Regime detection
- Reinforcement learning
- Portfolio optimization
- Backtesting engine
- Sentiment analysis integration
- Cross-asset modeling

---

# Disclaimer

This project is for research and educational purposes only.

It is not financial advice and should not be used directly for investment decisions without rigorous validation and risk management.

---