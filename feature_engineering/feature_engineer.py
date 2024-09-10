import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def engineer_features(sentiment_scores, stock_data):
    features = {}
    
    # Convert sentiment_scores to a Series if it's a DataFrame
    if isinstance(sentiment_scores, pd.DataFrame):
        if len(sentiment_scores.columns) == 1:
            sentiment_scores = sentiment_scores.iloc[:, 0]
        else:
            raise ValueError("sentiment_scores DataFrame must have only one column")
    
    # Ensure sentiment_scores is a pd.Series with a DatetimeIndex
    if not isinstance(sentiment_scores, pd.Series):
        raise ValueError("sentiment_scores must be a pandas Series or a single-column DataFrame with a DatetimeIndex.")
    
    # Ensure the index is a DatetimeIndex
    sentiment_scores.index = pd.to_datetime(sentiment_scores.index)
    
    # Aggregate sentiment scores to daily level
    daily_sentiment = sentiment_scores.resample('H').mean()
    
    logger.info(f"Daily sentiment scores range: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
    
    for symbol, data in stock_data.items():
        logger.info(f"Processing features for {symbol}")
        
        # Ensure the index is a DatetimeIndex and sort it
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data.sort_index()
        
        # Remove any duplicate indices
        data = data[~data.index.duplicated(keep='first')]
        
        logger.info(f"Stock data range for {symbol}: {data.index.min()} to {data.index.max()}")
        logger.info(f"Stock data columns: {data.columns}")
        logger.info(f"Stock data shape before processing: {data.shape}")
        
        # Find the overlapping date range
        start_date = max(data.index.min(), daily_sentiment.index.min())
        end_date = min(data.index.max(), daily_sentiment.index.max())
        
        logger.info(f"Overlapping range for {symbol}: {start_date} to {end_date}")
        
        # Filter both dataframes to the overlapping date range
        data = data.loc[start_date:end_date]
        sentiment_df = daily_sentiment.loc[start_date:end_date].to_frame(name='sentiment')
        
        logger.info(f"Stock data shape after date range filter: {data.shape}")
        logger.info(f"Sentiment data shape after date range filter: {sentiment_df.shape}")
        
        if data.empty or sentiment_df.empty:
            logger.warning(f"No overlapping data for {symbol}. Skipping this symbol.")
            continue
        
        # Combine stock data with sentiment scores
        combined_data = data.join(sentiment_df, how='left')
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Combined data columns: {combined_data.columns}")
        
        # Fill any remaining NaN values in sentiment column
        combined_data['sentiment'] = combined_data['sentiment'].fillna(method='ffill').fillna(0)
        
        # Calculate moving average of sentiment scores
        combined_data['SentimentMA5'] = combined_data['sentiment'].rolling(window=5).mean()
        
        # Calculate the difference between current sentiment and its moving average
        combined_data['SentimentDiff'] = combined_data['sentiment'] - combined_data['SentimentMA5']
        
        # Calculate the correlation between sentiment and returns
        combined_data['SentimentReturnsCorr'] = combined_data['sentiment'].rolling(window=5).corr(combined_data['Returns'])
        
        # Create lagged features
        for i in range(1, 6):
            combined_data[f'Returns_Lag_{i}'] = combined_data['Returns'].shift(i)
            combined_data[f'Sentiment_Lag_{i}'] = combined_data['sentiment'].shift(i)
        
        # Create target variable (next day's return)
        combined_data['Target'] = combined_data['Returns'].shift(-1)
        
        logger.info(f"Data shape before dropping NaNs: {combined_data.shape}")
        logger.info(f"Columns with NaNs: {combined_data.columns[combined_data.isna().any()].tolist()}")
        
        # Dropping all NaN values
        combined_data = combined_data.dropna()
        
        logger.info(f"Data shape after dropping NaNs: {combined_data.shape}")
        
        if combined_data.empty:
            logger.warning(f"After processing, no valid data remains for {symbol}. Skipping this symbol.")
            continue
        
        logger.info(f"Processed {len(combined_data)} rows of data for {symbol}")
        features[symbol] = combined_data
    
    if not features:
        logger.error("No valid data remained after feature engineering for any symbol.")
        return None  # Return None instead of raising an exception
    
    return features