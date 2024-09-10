import pandas as pd
import numpy as np

def prepare_stock_data(stock_data):
    for symbol, data in stock_data.items():
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Calculate moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Calculate relative strength index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        stock_data[symbol] = data
    
    return stock_data

