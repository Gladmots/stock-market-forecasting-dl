# forecast.py
from pathlib import Path
import numpy as np, pandas as pd, joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import tensorflow as tf

# Keep everything inside the package folder
ROOT = Path(__file__).resolve().parent                 # stock_market_forecasting/
DATA_PATH = ROOT / "stock_data.csv"                    # written by dataset.py
MODELS_DIR = ROOT / "models"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "stock_price_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH  = MODELS_DIR / "target_scaler.pkl"

SEQ_LEN_DEFAULT = 60  # must match training unless you pass different at train-time

def _hardened_load_csv(path: Path) -> pd.DataFrame:
    """Load CSV robustly and return a DataFrame with Date + Close (numeric)."""
    df = pd.read_csv(path)

    # If MultiIndex ever sneaks in, flatten (CSV usually gives flat cols)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in tup if str(c) != '']) for tup in df.columns]

    # Normalize names
    df.columns = [str(c).strip() for c in df.columns]

    # Date column
    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if not date_col:
        raise ValueError(f"No 'Date' column found. Columns: {df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Find Close / Adj Close
    close_candidates = [c for c in df.columns if c.lower() == "close" or c.lower().endswith("_close")]
    if not close_candidates:
        close_candidates = [c for c in df.columns if c.lower() in ("adj close", "adj_close") or c.lower().endswith("_adj close")]
    if not close_candidates:
        raise ValueError(f"No Close/Adj Close column found. Columns: {df.columns.tolist()}")

    close_col = close_candidates[0]

    # Keep Date + Close only, enforce numeric Close, sort, ffill
    df = df[[date_col, close_col]].rename(columns={close_col: "Close"})
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values(date_col).ffill().reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA50"]  = df["Close"].rolling(50, min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()
    return df

def create_sequences(X, y, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def predict_future(model, feature_scaler, target_scaler, df_feat: pd.DataFrame, seq_len: int, steps: int):
    feature_cols = ["Close", "MA50", "MA200"]
    hist = df_feat[feature_cols].copy()

    seq = feature_scaler.transform(hist.iloc[-seq_len:].values)
    preds, dates = [], []
    last_date = df_feat["Date"].iloc[-1]

    buffer_unscaled = hist.iloc[-seq_len:].copy()
    for _ in range(steps):
        pred_scaled = model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
        pred_close = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(pred_close)
        last_date = last_date + timedelta(days=1)
        dates.append(last_date)

        # append & recompute MAs
        buffer_unscaled = pd.concat(
            [buffer_unscaled, pd.DataFrame([{"Close": pred_close}], index=[last_date])]
        )
        buffer_unscaled["MA50"]  = buffer_unscaled["Close"].rolling(50, min_periods=1).mean()
        buffer_unscaled["MA200"] = buffer_unscaled["Close"].rolling(200, min_periods=1).mean()
        buffer_unscaled = buffer_unscaled.iloc[-seq_len:]
        seq = feature_scaler.transform(buffer_unscaled[feature_cols].values)

    return pd.Series(preds, index=pd.to_datetime(dates), name="Predicted")

def main(horizon: int = 30, seq_len: int = SEQ_LEN_DEFAULT):
    if not DATA_PATH.exists(): raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = _hardened_load_csv(DATA_PATH)
    df = make_features(df)

    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler  = joblib.load(TARGET_SCALER_PATH)
    model = load_model(MODEL_PATH)

    feature_cols = ["Close", "MA50", "MA200"]
    target_col   = "Close"

    X_all = feature_scaler.transform(df[feature_cols].values)
    y_all = target_scaler.transform(df[[target_col]].values)

    X_seq, y_seq = create_sequences(X_all, y_all, seq_len)
    split_idx = int(len(X_seq) * 0.8)
    X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]

    val_pred_scaled = model.predict(X_val, verbose=0)
    val_pred = target_scaler.inverse_transform(val_pred_scaled)
    y_val_inv = target_scaler.inverse_transform(y_val)

    # aligned dates: first y corresponds to index seq_len
    val_start_idx = split_idx + seq_len
    val_dates = df["Date"].iloc[val_start_idx:val_start_idx+len(y_val)].to_numpy()

    # metrics
    rmse = float(np.sqrt(mean_squared_error(y_val_inv[:, 0], val_pred[:, 0])))  # no 'squared' kw
    mae  = float(mean_absolute_error(y_val_inv[:, 0], val_pred[:, 0]))
    mape = float(np.mean(np.abs((y_val_inv[:, 0] - val_pred[:, 0]) / np.maximum(1e-8, y_val_inv[:, 0]))) * 100)


    with open(OUT / "metrics.json", "w") as f:
        json.dump({"RMSE": rmse, "MAE": mae, "MAPE_%": mape}, f, indent=2)

    # save validation table
    pd.DataFrame({
        "Date": pd.to_datetime(val_dates),
        "Actual_Close": y_val_inv[:,0],
        "Pred_Close":   val_pred[:,0],
    }).to_csv(OUT / "val_predictions.csv", index=False)

    # plots
    plt.figure(figsize=(12,5))
    plt.plot(val_dates, y_val_inv[:,0], label="Actual Close")
    plt.plot(val_dates, val_pred[:,0], label="Predicted Close")
    plt.title("Validation: Actual vs Predicted")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "val_plot.png", dpi=150); plt.close()

    # future
    df_feat = df[["Date","Close"]].assign(
        MA50=df["Close"].rolling(50, min_periods=1).mean(),
        MA200=df["Close"].rolling(200, min_periods=1).mean(),
    )
    future = predict_future(model, feature_scaler, target_scaler, df_feat, seq_len, steps=horizon)
    future.to_frame().reset_index(names="Date").to_csv(OUT / "future_predictions.csv", index=False)
    
    # plot future
    hist_tail = df.tail(120)
    plt.figure(figsize=(12,5))
    plt.plot(hist_tail["Date"], hist_tail["Close"], label="Historical Close")
    plt.plot(future.index, future.values, label=f"Future ({horizon} steps)")
    plt.title("Next Forecast"); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "future_plot.png", dpi=150); plt.close()

    print(f"[forecast] Saved metrics & plots in: {OUT}")

if __name__ == "__main__":
    main()
