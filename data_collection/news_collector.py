import requests
import pandas as pd
from datetime import datetime, timedelta

def collect_news(num_articles=100, api_key=None):
    if api_key is None:
        raise ValueError("Please provide a valid NewsAPI key")

    base_url = "https://newsapi.org/v2/everything"
    
    # Calculate date range (last 30 days)
    end_date = datetime.now(pd.Timestamp.utcnow().tzinfo)
    start_date = end_date - timedelta(days=30)
    
    params = {
        'q': 'stock market OR finance OR economy',
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': min(num_articles, 100),  # API limit is 100 per request
        'apiKey': api_key,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        articles = data.get('articles', [])
        
        news_data = [{
            'source': article['source']['name'],
            'title': article['title'],
            'content': article['description'] or article['content'],
            'date': article['publishedAt']
        } for article in articles]

        df = pd.DataFrame(news_data)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        print(f"Collected {len(df)} news articles.")
        return df

    except requests.RequestException as e:
        print(f"An error occurred while fetching news: {str(e)}")
        return pd.DataFrame(columns=['source', 'title', 'content', 'date'])

if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual NewsAPI key
    api_key = '32e0bf3cf9554d36ad0fbbaef516b68c'
    print(collect_news(5, api_key))