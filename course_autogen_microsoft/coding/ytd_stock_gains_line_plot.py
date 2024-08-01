# filename: ytd_stock_gains_line_plot.py

import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import time

# Define the stock tickers
tickers = ["NVDA", "TSLA"]

# Define the date range
start_date = "2024-01-01"
end_date = "2024-06-30"

# Function to fetch stock data with retries
def fetch_stock_data(tickers, start_date, end_date, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
            return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
            attempt += 1
    raise Exception(f"Failed to fetch stock data after {retries} attempts")

# Fetch the stock data
data = fetch_stock_data(tickers, start_date, end_date)

# Calculate the cumulative returns
cumulative_returns = (data / data.iloc[0] - 1) * 100

# Plot the cumulative YTD returns
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)

# Add title and labels
plt.title("Cumulative YTD Stock Gains for NVDA and TSLA", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative YTD Gain (%)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig("ytd_stock_gains.png")
plt.show()