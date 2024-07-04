import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define the path to the data file
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.csv')

# Check if the data file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")

# Load the data
data = pd.read_csv(data_path)

# Load the model
model = load_model('../models/stock_price_model.h5')

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 60
X, y = create_sequences(scaled_data.values, seq_length)

# Predict on the validation set
predictions = model.predict(X[-len(y):])

# Inverse transform predictions
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y):], scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Price')
plt.plot(data.index[-len(y):], predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Predict future stock prices
def predict_future_prices(model, data, seq_length, steps):
    predictions = []
    current_sequence = data[-seq_length:]

    for _ in range(steps):
        prediction = model.predict(current_sequence[np.newaxis, :, :])
        predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Predict the next 30 days
future_predictions = predict_future_prices(model, scaled_data.values, seq_length, 30)

# Plot the future predictions
plt.figure(figsize=(10, 6))
plt.plot(data.index[-60:], scaler.inverse_transform(scaled_data.values[-60:, 0].reshape(-1, 1)), label='Historical Prices')
plt.plot(pd.date_range(start=data.index[-1], periods=31, closed='right'), future_predictions, label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
