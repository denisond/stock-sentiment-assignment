import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_stock_prices(symbols=['AAPL', 'GOOGL', 'MSFT'], days=365):
    stock_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not data.empty:
                stock_data[symbol] = data
                print(f"Collected {len(data)} days of stock data for {symbol}.")
            else:
                print(f"No data available for {symbol}.")
        except Exception as e:
            print(f"Error collecting data for {symbol}: {str(e)}")
    
    return stock_data

if __name__ == "__main__":
    print(collect_stock_prices(symbols=['AAPL'], days=30))