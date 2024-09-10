from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text_data):
    if isinstance(text_data, pd.Series):
        return text_data.apply(get_sentiment)
    elif isinstance(text_data, list):
        return [get_sentiment(text) for text in text_data]
    else:
        return get_sentiment(text_data)

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

