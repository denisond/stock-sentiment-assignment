import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def engineer_features(sentiment_scores, stock_data):
    features = {}
    
    for symbol, data in stock_data.items():
        logger.info(f"Processing features for {symbol}")
        
        # Ensure the index is a DatetimeIndex, sort it, and make it tz-naive
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data.sort_index()
        
        # Remove any duplicate indices
        data = data[~data.index.duplicated(keep='first')]
        
        # Create a DataFrame for sentiment scores and make it tz-naive
        sentiment_df = pd.DataFrame({'sentiment': sentiment_scores}, index=pd.to_datetime(sentiment_scores.index).tz_localize(None))
        sentiment_df = sentiment_df.sort_index()
        
        # Find the overlapping date range
        start_date = max(data.index.min(), sentiment_df.index.min())
        end_date = min(data.index.max(), sentiment_df.index.max())
        
        logger.info(f"Stock data range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Sentiment data range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")
        logger.info(f"Overlapping range: {start_date} to {end_date}")
        
        # Filter both dataframes to the overlapping date range
        data = data.loc[start_date:end_date]
        sentiment_df = sentiment_df.loc[start_date:end_date]
        
        if data.empty or sentiment_df.empty:
            logger.warning(f"No overlapping data for {symbol}. Skipping this symbol.")
            continue
        
        # Reindex sentiment data to match stock data index, forward filling missing values
        sentiment_df = sentiment_df.reindex(data.index, method='ffill')
        
        # Combine stock data with sentiment scores
        combined_data = data.join(sentiment_df, how='left')
        
        # Fill any remaining NaN values in sentiment column
        combined_data['sentiment'] = combined_data['sentiment'].fillna(method='ffill').fillna(0)
        
        # Calculate moving average of sentiment scores
        combined_data['SentimentMA5'] = combined_data['sentiment'].rolling(window=5).mean()
        
        # Calculate the difference between current sentiment and its moving average
        combined_data['SentimentDiff'] = combined_data['sentiment'] - combined_data['SentimentMA5']
        
        # Calculate the correlation between sentiment and returns
        combined_data['SentimentReturnsCorr'] = combined_data['sentiment'].rolling(window=20).corr(combined_data['Returns'])
        
        # Create lagged features
        for i in range(1, 6):
            combined_data[f'Returns_Lag_{i}'] = combined_data['Returns'].shift(i)
            combined_data[f'Sentiment_Lag_{i}'] = combined_data['sentiment'].shift(i)
        
        # Create target variable (next day's return)
        combined_data['Target'] = combined_data['Returns'].shift(-1)
        
        # Drop NaN values
        combined_data.dropna(inplace=True)
        
        if combined_data.empty:
            logger.warning(f"After processing, no valid data remains for {symbol}. Skipping this symbol.")
            continue
        
        logger.info(f"Processed {len(combined_data)} rows of data for {symbol}")
        features[symbol] = combined_data
    
    if not features:
        raise ValueError("No valid data remained after feature engineering for any symbol.")
    
    return features

if __name__ == "__main__":
    # This is just a placeholder to demonstrate how the function would be used
    import yfinance as yf
    from data_preprocessing.sentiment_analyzer import analyze_sentiment
    
    # Get some sample stock data
    stock_data = {'AAPL': yf.download('AAPL', start='2020-01-01', end='2023-01-01')}
    
    # Generate some random sentiment scores (with a different date range)
    date_range = pd.date_range(start='2019-06-01', end='2022-12-31')
    sentiment_scores = pd.Series(np.random.randn(len(date_range)), index=date_range)
    
    features = engineer_features(sentiment_scores, stock_data)
    for symbol, data in features.items():
        print(f"{symbol}:")
        print(data.head())
        print(f"Shape: {data.shape}")