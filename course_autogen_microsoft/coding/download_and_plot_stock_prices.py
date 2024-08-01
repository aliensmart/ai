# filename: download_and_plot_stock_prices.py

from functions import get_stock_prices, plot_stock_prices
import pandas as pd

# Define the stock symbols and date range
stock_symbols = ['NVDA', 'TSLA']
start_date = '2024-01-01'
end_date = '2024-06-30'

# Get the stock prices for the given symbols and date range
stock_prices = get_stock_prices(stock_symbols, start_date, end_date)

# Plot the stock prices and save the figure to a file
plot_stock_prices(stock_prices, 'stock_prices_YTD_plot.png')

print("Stock prices plotted and saved to stock_prices_YTD_plot.png")