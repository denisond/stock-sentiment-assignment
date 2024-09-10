import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json

def collect_news(num_articles=10000, api_key=None, save_path='data/news_articles'):
    if api_key is None:
        raise ValueError("Please provide a valid NewsAPI key")

    base_url = "https://newsapi.org/v2/everything"
    
    # Calculate date range (last 30 days for free plan)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Free plan allows up to 1 month in the past

    # Prepare to store all articles
    all_articles = []

    # Loop through each day in the date range
    current_date = start_date
    while current_date <= end_date:
        # Construct the file path for each date range
        file_path = os.path.join(save_path, f"news_articles_{current_date.strftime('%Y%m%d')}.json")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Loading from saved file.")
            with open(file_path, 'r') as f:
                daily_articles = json.load(f)
            all_articles.extend(daily_articles)
        else:
            # Fetch the news from the API for the current date
            page = 1
            while True:
                params = {
                    'q': 'stock market OR finance OR economy',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 100,  # API limit is 100 per request
                    'page': page,
                    'apiKey': api_key,
                    'from': current_date.strftime('%Y-%m-%d'),
                    'to': (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
                }

                try:
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()  # Raise an exception for bad status codes

                    if response.status_code == 426:
                        print("Upgrade required. Please check the API documentation.")
                        return pd.DataFrame(columns=['source', 'title', 'content', 'date', 'url'])

                    data = response.json()

                    articles = data.get('articles', [])
                    
                    if not articles:
                        break  # No more articles, exit the page loop
                    
                    daily_articles = [{
                        'source': article['source']['name'],
                        'title': article['title'],
                        'content': article['description'] or article['content'],
                        'date': article['publishedAt'],
                        'url': article['url']
                    } for article in articles]

                    # Save articles for the current day
                    with open(file_path, 'w') as f:
                        json.dump(daily_articles, f, indent=4)
                    print(f"Saved articles to {file_path}")

                    all_articles.extend(daily_articles)
                    page += 1  # Move to the next page

                except requests.RequestException as e:
                    print(f"An error occurred while fetching news: {str(e)}")
                    break
        
        current_date += timedelta(days=1)  # Move to the next day

    # Convert to DataFrame and return
    df = pd.DataFrame(all_articles)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    print(f"Collected {len(df)} news articles.")
    return df
