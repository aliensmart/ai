# filename: ytd_stock_gains_plot.py

import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

# Define the stock tickers
tickers = ["NVDA", "TSLA"]

# Define the date range
start_date = "2024-01-01"
end_date = "2024-06-30"

# Fetch the stock data
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# Calculate the YTD percentage gain
ytd_gains = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100

# Plot the YTD gains
plt.figure(figsize=(10, 6))
ytd_gains.plot(kind='bar', color=['blue', 'green'])
plt.title("YTD Stock Gains for NVDA and TSLA")
plt.xlabel("Stock Ticker")
plt.ylabel("YTD Gain (%)")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig("ytd_stock_gains.png")
plt.show()