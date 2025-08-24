# src/cli.py
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse

from . import dataset as dataset_mod
from . import model as model_mod
from . import forecast as forecast_mod
from . import report as report_mod 

from pathlib import Path

ROOT = Path(__file__).resolve().parent         # <-- was parent.parent
DATA_PATH = ROOT / "stock_data.csv"            # <-- was ROOT / "data" / "stock_data.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "stock_price_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH  = MODELS_DIR / "target_scaler.pkl"
HISTORY_PATH = MODELS_DIR / "history.json"


def main():
    parser = argparse.ArgumentParser(
        prog="stock-forecasting",
        description="Download data, train model, forecast prices, and build a PDF report."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download data
    p_dl = sub.add_parser("download", help="Download historical OHLCV data")
    p_dl.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker")
    p_dl.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p_dl.add_argument("--end", default="2024-01-01", help="End date (YYYY-MM-DD)")

    # train data
    p_tr = sub.add_parser("train", help="Train LSTM model")
    p_tr.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    p_tr.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_tr.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # forecast
    p_fc = sub.add_parser("forecast", help="Validate and forecast future prices")
    p_fc.add_argument("--horizon", type=int, default=30, help="Forecast steps (days)")
    p_fc.add_argument("--no-report", action="store_true",
                      help="Skip auto-generating the PDF report")

    # report data
    sub.add_parser("report", help="Generate PDF report from outputs")

    args = parser.parse_args()

    if args.command == "download":
        dataset_mod.main(ticker=args.ticker, start=args.start, end=args.end)

    elif args.command == "train":
        model_mod.main(seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size)

    elif args.command == "forecast":
        forecast_mod.main(horizon=args.horizon)
        if not args.no_report:
            report_mod.main()

    elif args.command == "report":
        report_mod.main()

if __name__ == "__main__":
    main()
