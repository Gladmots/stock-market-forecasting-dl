import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../data/stock_data.csv')
data.fillna(method='ffill', inplace=True)

# Create additional features (e.g., moving averages)
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close', 'MA50', 'MA200']])

# Convert scaled data to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=['Close', 'MA50', 'MA200'])

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

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, scaled_data.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('../models/stock_price_model.h5')
print("Model training complete. Model saved to models/stock_price_model.h5")

# Make predictions
predictions = model.predict(X_val)
predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], 2))], axis=1))[:, 0]

# Visualize the data
plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Actual Price')
plt.plot(range(len(data) - len(predictions), len(data)), predictions, label='LSTM Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price and LSTM Predictions')
plt.show()

# Implement and visualize Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
linear_predictions = linear_model.predict(X_val.reshape(X_val.shape[0], -1))
linear_predictions = scaler.inverse_transform(np.concatenate([linear_predictions, np.zeros((linear_predictions.shape[0], 2))], axis=1))[:, 0]

plt.figure(figsize=(14, 5))
plt.plot(data['Close'], label='Actual Price')
plt.plot(range(len(data) - len(linear_predictions), len(data)), linear_predictions, label='Linear Regression Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price and Linear Regression Predictions')
plt.show()
