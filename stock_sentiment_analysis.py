import os
import logging
from dotenv import load_dotenv
from data_collection import news_collector, social_media_collector, stock_price_collector
from data_preprocessing import text_cleaner, sentiment_analyzer, stock_data_preparer
from feature_engineering import feature_engineer
from model_development import sentiment_model, stock_prediction_model
from model_evaluation import evaluate_model
from visualization import data_visualizer
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_utc(dt):
    if dt.tzinfo is None:
        return pd.Timestamp(dt).tz_localize('UTC')
    return pd.Timestamp(dt).tz_convert('UTC')

def main():
    try:
        # Get the NewsAPI key from environment variable
        news_api_key = os.getenv('NEWS_API_KEY')
        if not news_api_key:
            raise ValueError("NEWS_API_KEY environment variable is not set. Please set it in your .env file and try again.")

        # Data Collection
        logger.info("Collecting data...")
        news_data = news_collector.collect_news(api_key=news_api_key)
        if news_data.empty:
            raise ValueError("Failed to collect news data")
        news_data['date'] = news_data['date'].apply(convert_to_utc)
        logger.info(f"Collected {len(news_data)} news articles")
        logger.info(f"News data date range: {news_data['date'].min()} to {news_data['date'].max()}")
        
        social_media_data = social_media_collector.collect_social_media_data()
        if social_media_data.empty:
            raise ValueError("Failed to collect social media data")
        social_media_data['created_at'] = social_media_data['created_at'].apply(convert_to_utc)
        logger.info(f"Collected {len(social_media_data)} social media posts")
        logger.info(f"Social media data date range: {social_media_data['created_at'].min()} to {social_media_data['created_at'].max()}")
        
        stock_data = stock_price_collector.collect_stock_prices()
        if not stock_data:
            raise ValueError("Failed to collect stock price data")
        for symbol, data in stock_data.items():
            data.index = data.index.tz_localize('UTC')
            logger.info(f"Collected stock data for {symbol}: {len(data)} days")
            logger.info(f"Stock data date range for {symbol}: {data.index.min()} to {data.index.max()}")

        # Data Preprocessing
        logger.info("Preprocessing data...")
        text_data = pd.concat([news_data['content'], social_media_data['text']])
        cleaned_text_data = text_cleaner.clean_text(text_data)
        sentiment_scores = sentiment_analyzer.analyze_sentiment(cleaned_text_data)
        prepared_stock_data = stock_data_preparer.prepare_stock_data(stock_data)

        # Ensure sentiment_scores have a datetime index and make it tz-naive
        sentiment_scores.index = pd.to_datetime(news_data['date'].tolist() + social_media_data['created_at'].tolist())
        sentiment_scores.index = sentiment_scores.index.tz_convert(None)
        sentiment_scores = sentiment_scores.sort_index()
        logger.info(f"Sentiment scores date range: {sentiment_scores.index.min()} to {sentiment_scores.index.max()}")
        logger.info(f"Number of sentiment scores: {len(sentiment_scores)}")

        # Feature Engineering
        logger.info("Engineering features...")
        features = feature_engineer.engineer_features(sentiment_scores, prepared_stock_data)

        # Model Development
        logger.info("Developing models...")
        news_data['sentiment'] = sentiment_analyzer.analyze_sentiment(news_data['content'])
        sentiment_model_results = sentiment_model.train_and_evaluate(news_data)
        stock_prediction_results = stock_prediction_model.train_and_evaluate(features)

        # Model Evaluation
        logger.info("Evaluating models...")
        evaluate_model.evaluate(sentiment_model_results, stock_prediction_results)

        # Visualization
        logger.info("Generating visualizations...")
        for symbol in features.keys():
            data_visualizer.plot_stock_prediction(features, stock_prediction_results[symbol]['model'], symbol)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()