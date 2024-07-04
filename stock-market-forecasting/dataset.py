import yfinance as yf
import pandas as pd

# Download historical data for Apple Inc.
data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')

# Save the data to a CSV file
data.to_csv('../data/stock_data.csv')

print("Data collection complete. Data saved to data/stock_data.csv")
