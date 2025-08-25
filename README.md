# 📈 Stock Market Forecasting with Deep Learning
A clean, end-to-end time-series pipeline for forecasting stock prices with deep learning (LSTM / GRU).  
It downloads market data, engineers features, trains a model, evaluates results, makes future forecasts, saves charts/tables, and builds a one-page PDF report.

![Stock forecasting screenshot](https://raw.githubusercontent.com/Gladmots/stock-market-forecasting-dl/main/stock_market_forecasting/Screenshot%202025-08-24%20160607.png)

![Stock forecasting screenshot 2](https://raw.githubusercontent.com/Gladmots/stock-market-forecasting-dl/main/stock_market_forecasting/Screenshot%202025-08-24%20160629.png)

## Features
- **Data ingest** from Yahoo Finance (OHLCV)
- **Feature engineering** (e.g., MA50 / MA200)
- **Deep learning** models: LSTM (default) and GRU (flag)
- **Chronological** train/val/test splits
- **Metrics:** RMSE, MAE, R² (+ simple backtest: CAGR, Max Drawdown, Sharpe)
- **Visuals:** Actual vs Predicted (validation) & N-day **future forecast**
- **Automated report:** One-pager **PDF** with charts + metrics
- **Single CLI** to run the whole process


> If you are running directly from `src/`, use `python -m src.cli ...`.  
> If installed as a package, use `python -m stock_market_forecasting.cli ...`.

## Prerequisites & Setup

- **Python** 3.9+  
- **pip** (and optionally a virtual environment)

## Create & activate a virtual environment:


## Windows (PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1

## macOS / Linux
python -m venv .venv
source .venv/bin/activate

## Install dependencies:

pip install -r requirements.txt

## Start Program

## Download data

python -m stock_market_forecasting.cli download \
  --ticker AAPL --start 2010-01-01 --end 2024-01-01


Creates a cleaned dataset under data/processed/.

## Train a model

python -m stock_market_forecasting.cli train \
  --ticker AAPL --seq-len 60 --epochs 60 --batch-size 32 --model lstm


Saves weights, scaler, and training history to models/.

## Forecast & evaluate

python -m stock_market_forecasting.cli forecast \
  --ticker AAPL --horizon 30

## Outputs

1. CSV
2. Images
3. Metrics
4. Report

> ⚠️ **Disclaimer:** Educational project only — **not** financial advice.

