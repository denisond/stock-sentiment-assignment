import pandas as pd
import numpy as np

def prepare_stock_data(stock_data):
    for symbol, data in stock_data.items():
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Calculate moving averages
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate relative strength index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        stock_data[symbol] = data
    
    return stock_data

if __name__ == "__main__":
    import yfinance as yf
    sample_data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
    prepared_data = prepare_stock_data({'AAPL': sample_data})
    print(prepared_data['AAPL'].head())
