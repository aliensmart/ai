# filename: analyze_nvda_stock_data.py
import pandas as pd

# Load the stock data from CSV file
stock_data = pd.read_csv('nvidia_stock_data.csv', index_col='Date', parse_dates=True)

# Calculate daily returns
stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()

# Describe the main statistics
description = stock_data.describe()

# Print summary statistics
print(description)

# Plot closing prices and daily returns for visual analysis
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))

# Plot Closing Prices
plt.subplot(2, 1, 1)
plt.plot(stock_data['Adj Close'])
plt.title('Nvidia Stock Adjusted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')

# Plot Daily Returns
plt.subplot(2, 1, 2)
plt.plot(stock_data['Daily Return'])
plt.title('Nvidia Stock Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')

plt.tight_layout()
plt.show()