# filename: ytd_stock_gains_plot_v2.py

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
bars = plt.bar(ytd_gains.index, ytd_gains.values, color=['skyblue', 'salmon'])

# Add title and labels
plt.title("YTD Stock Gains for NVDA and TSLA", fontsize=14, fontweight='bold')
plt.xlabel("Stock Ticker", fontsize=12)
plt.ylabel("YTD Gain (%)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the data values on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', va='bottom', ha='center', fontsize=12, fontweight='bold', color='black')

plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot to a file
plt.savefig("ytd_stock_gains.png")
plt.show()