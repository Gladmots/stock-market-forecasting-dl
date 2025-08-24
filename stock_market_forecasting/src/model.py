# model.py
# model.py
import os, random
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf, joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Reproducibility
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0); tf.random.set_seed(0)

# Paths (keep everything inside the package folder)
ROOT = Path(__file__).resolve().parent            # stock_market_forecasting/
DATA_PATH = ROOT / "stock_data.csv"               # written by dataset.py

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "stock_price_model.keras"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH  = MODELS_DIR / "target_scaler.pkl"
HISTORY_PATH        = MODELS_DIR / "history.json"

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

def main(seq_len: int = 60, epochs: int = 100, batch_size: int = 32):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    # ---------- Hardened CSV load ----------
    df = pd.read_csv(DATA_PATH)

    # If CSV came from multi-ticker download, columns might be MultiIndex once upon a time
    # (but CSV read gives flat columns). Keep this for completeness:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in tup if str(c) != '']) for tup in df.columns]

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize date column
    date_col = "Date" if "Date" in df.columns else "date"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).ffill()
    else:
        raise ValueError("No 'Date' column found in CSV.")

    # Find a proper closing price column
    close_candidates = [c for c in df.columns if c.lower() == "close" or c.lower().endswith("_close")]
    if not close_candidates:
        # yfinance sometimes writes 'Adj Close'
        close_candidates = [c for c in df.columns if c.lower() in ("adj close", "adj_close") or c.lower().endswith("_adj close")]
    if not close_candidates:
        raise ValueError(f"No Close/Adj Close column found. Columns: {df.columns.tolist()}")

    close_col = close_candidates[0]

    # Keep only what we need and enforce numeric
    df = df[[date_col, close_col]].rename(columns={close_col: "Close"})
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    # ---------------------------------------

    df = make_features(df)

    feature_cols = ["Close", "MA50", "MA200"]
    target_col   = "Close"

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_all = feature_scaler.fit_transform(df[feature_cols].values)
    y_all = target_scaler.fit_transform(df[[target_col]].values)

    X_seq, y_seq = create_sequences(X_all, y_all, seq_len)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=1)

    model.save(MODEL_PATH)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)

    import json
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)

    print(f"[train] Model:   {MODEL_PATH}")
    print(f"[train] Scalers: {FEATURE_SCALER_PATH}, {TARGET_SCALER_PATH}")
    print(f"[train] Done. seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}")

if __name__ == "__main__":
    main()
