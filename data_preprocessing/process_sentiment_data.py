import pandas as pd


def process_sentiment_data(news_data, social_media_data, sentiment_analysis_scores):
    # Combine dates from news_data and social_media_data into a single list
    sentiment_dates = pd.to_datetime(news_data['date'].tolist() + social_media_data['created_at'].tolist())

    # Create a DataFrame to combine the dates and the sentiment scores
    sentiment_df = pd.DataFrame({
        'date': sentiment_dates,
        'score': sentiment_analysis_scores
    })

    # Drop duplicate rows based on the 'date' column, keeping the first occurrence
    sentiment_df = sentiment_df.drop_duplicates(subset='date', keep='first')

    # Set 'date' as the index
    sentiment_df.set_index('date', inplace=True)
    sentiment_df.index = sentiment_df.index.tz_localize(None)
    sentiment_df = sentiment_df.sort_index()

    # Filter sentiment_scores to only include entries from the last 30 days
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(days=30)
    sentiment_scores_filtered = sentiment_df.loc[sentiment_df.index >= cutoff_date]

    return sentiment_scores_filtered


