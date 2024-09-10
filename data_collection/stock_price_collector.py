import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_stock_prices(symbols=['^GSPC']):
    stock_data = {}
    
    # Use UTC for both start and end dates, ensuring consistency with yfinance data
    end_date = datetime.now().astimezone(pd.Timestamp.utcnow().tz)  # Ensuring timezone-aware
    start_date = end_date - timedelta(days=100)  # Adjust this as needed

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1h'  # Use hourly intervals
            )
            if not data.empty:
                # Ensure the data is timezone-aware and matches UTC
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                else:
                    data.index = data.index.tz_convert('UTC')
                
                # Resample to hourly data, keeping the first entry for each hour
                data = data.resample('H', label='left', closed='left').first().dropna()
                
                stock_data[symbol] = data
                print(f"Collected {len(data)} hours of stock data for {symbol}.")
                print(f"Data range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"No data available for {symbol}.")
        except Exception as e:
            print(f"Error collecting data for {symbol}: {str(e)}")
    
    return stock_data
