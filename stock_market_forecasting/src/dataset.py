# dataset.py
# dataset.py
import yfinance as yf
import pandas as pd
from pathlib import Path

# Save next to the package modules (matches model.py)
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "stock_data.csv"

def main(ticker: str = "AAPL", start: str = "2010-01-01", end: str = "2024-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Downloaded empty DataFrame for {ticker}. Check ticker/dates.")
    df.reset_index(inplace=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[download] Saved: {CSV_PATH} ({len(df):,} rows)")

if __name__ == "__main__":
    main()

