import pandas as pd
import random
from datetime import datetime, timedelta

def collect_social_media_data(num_posts=1000):
    social_media_data = []
    topics = ['#stocks', '#investing', '#finance', '#economy']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for _ in range(num_posts):
        random_date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
        social_media_data.append({
            'text': f"This is a sample post about {random.choice(topics)}",
            'created_at': random_date,
            'user': f"user_{random.randint(1000, 9999)}",
            'retweets': random.randint(0, 100),
            'likes': random.randint(0, 500)
        })
    
    df = pd.DataFrame(social_media_data)
    print(f"Collected {len(df)} social media posts.")
    return df

if __name__ == "__main__":
    print(collect_social_media_data(5))