# filename: fetch_nvda_stock_data.py
import yfinance as yf
import pandas as pd

# Define the stock ticker symbol for Nvidia
ticker = 'NVDA'

# Define the date range for the past month
end_date = '2024-04-23'
start_date = pd.to_datetime(end_date) - pd.DateOffset(months=1)
start_date = start_date.strftime('%Y-%m-%d')

# Download stock price data for the past month
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data for verification
print(stock_data.head())

# Save the data to a CSV file for convenience
stock_data.to_csv('nvidia_stock_data.csv')