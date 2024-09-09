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

if __name__ == "__main__":
    sample_text = "The company's earnings report exceeded expectations, leading to a surge in stock price."
    print(analyze_sentiment(sample_text))
    
    sample_series = pd.Series(["Positive sentiment text.", "Negative sentiment text."])
    print(analyze_sentiment(sample_series))