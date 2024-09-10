# import praw
# import pandas as pd
# from datetime import datetime, timedelta
# import os
# import json
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def collect_social_media_data(save_path='social_media_posts', max_posts_per_subreddit_per_day=50, max_total_posts_per_day=200):
#     # Reddit API credentials
#     client_id = os.getenv('REDDIT_CLIENT_ID')
#     client_secret = os.getenv('REDDIT_CLIENT_SECRET')
#     user_agent = os.getenv('REDDIT_USER_AGENT', 'MyBot/1.0')

#     if not all([client_id, client_secret]):
#         raise ValueError("Reddit API credentials are not set. Please set them in your environment variables.")

#     # Set up search parameters
#     subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
#     end_date = datetime.utcnow()
#     start_date = end_date - timedelta(days=30)

#     # Prepare to store all posts
#     all_posts = []

#     # Loop through each day in the date range
#     current_date = start_date
#     while current_date <= end_date:
#         logger.info(f"Fetching data for {current_date.strftime('%Y-%m-%d')}")
        
#         try:
#             reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
#             daily_posts = []  # Store posts for the current day across all subreddits

#             for subreddit_name in subreddits:
#                 subreddit_posts = []  # Store posts for this subreddit for the current day
#                 subreddit = reddit.subreddit(subreddit_name)
#                 submissions = subreddit.new(limit=None)

#                 for submission in submissions:
#                     submission_date = datetime.fromtimestamp(submission.created_utc)
#                     if current_date <= submission_date < current_date + timedelta(days=1):
#                         subreddit_posts.append({
#                             'text': submission.title + " " + (submission.selftext if submission.selftext else ""),
#                             'created_at': submission_date.isoformat(),
#                             'user': str(submission.author),
#                             'upvotes': submission.score,
#                             'num_comments': submission.num_comments,
#                             'subreddit': subreddit_name
#                         })

#                     # Stop collecting for this subreddit once max posts are reached
#                     if len(subreddit_posts) >= max_posts_per_subreddit_per_day:
#                         break
                
#                 logger.info(f"Collected {len(subreddit_posts)} posts from r/{subreddit_name} for {current_date.strftime('%Y-%m-%d')}")
#                 daily_posts.extend(subreddit_posts)

#                 # Stop collecting across all subreddits if the total limit for the day is reached
#                 if len(daily_posts) >= max_total_posts_per_day:
#                     logger.info(f"Reached daily post limit of {max_total_posts_per_day} for {current_date.strftime('%Y-%m-%d')}")
#                     break

#             all_posts.extend(daily_posts)

#         except Exception as e:
#             logger.error(f"An error occurred while fetching Reddit posts: {str(e)}")
        
#         current_date += timedelta(days=1)  # Move to the next day

#     # Convert to DataFrame and save to a single file
#     df = pd.DataFrame(all_posts)
#     if not df.empty:
#         df['created_at'] = pd.to_datetime(df['created_at'])

#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         file_path = os.path.join(save_path, f"reddit_posts_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json")
#         with open(file_path, 'w') as f:
#             json.dump(all_posts, f, indent=4)
#         logger.info(f"Saved Reddit posts to {file_path}")

#     return df

# if __name__ == "__main__":
#     df = collect_social_media_data()
#     print(df.head())
#     print(f"Total Reddit posts collected: {len(df)}")


import praw
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_social_media_data(save_path='data/social_media_posts', max_posts_per_subreddit_per_day=50, max_total_posts_per_day=200):
    # Reddit API credentials
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'MyBot/1.0')

    if not all([client_id, client_secret]):
        raise ValueError("Reddit API credentials are not set. Please set them in your environment variables.")

    # Set up search parameters
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    # Prepare to store all posts across all days
    all_days_data = []

    # Loop through each day in the date range
    current_date = start_date
    while current_date <= end_date:
        daily_posts = []  # Store posts for the current day across all subreddits
        day_str = current_date.strftime('%Y%m%d')
        file_path = os.path.join(save_path, f"reddit_posts_{day_str}.json")
        
        # Check if the file for this day already exists
        if os.path.exists(file_path):
            logger.info(f"File for {current_date.strftime('%Y-%m-%d')} already exists, loading from {file_path}")
            with open(file_path, 'r') as f:
                daily_posts = json.load(f)
        else:
            logger.info(f"Fetching data for {current_date.strftime('%Y-%m-%d')}")

            try:
                reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

                for subreddit_name in subreddits:
                    subreddit_posts = []  # Store posts for this subreddit for the current day
                    subreddit = reddit.subreddit(subreddit_name)
                    submissions = subreddit.new(limit=None)

                    for submission in submissions:
                        submission_date = datetime.fromtimestamp(submission.created_utc)
                        if current_date <= submission_date < current_date + timedelta(days=1):
                            subreddit_posts.append({
                                'text': submission.title + " " + (submission.selftext if submission.selftext else ""),
                                'created_at': submission_date.isoformat(),
                                'user': str(submission.author),
                                'upvotes': submission.score,
                                'num_comments': submission.num_comments,
                                'subreddit': subreddit_name
                            })

                        # Stop collecting for this subreddit once max posts are reached
                        if len(subreddit_posts) >= max_posts_per_subreddit_per_day:
                            break

                    logger.info(f"Collected {len(subreddit_posts)} posts from r/{subreddit_name} for {current_date.strftime('%Y-%m-%d')}")
                    daily_posts.extend(subreddit_posts)

                    # Stop collecting across all subreddits if the total limit for the day is reached
                    if len(daily_posts) >= max_total_posts_per_day:
                        logger.info(f"Reached daily post limit of {max_total_posts_per_day} for {current_date.strftime('%Y-%m-%d')}")
                        break

                # Save posts for this day to a file
                if daily_posts:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    with open(file_path, 'w') as f:
                        json.dump(daily_posts, f, indent=4)
                    logger.info(f"Saved Reddit posts to {file_path}")

            except Exception as e:
                logger.error(f"An error occurred while fetching Reddit posts: {str(e)}")

        # Append the current day's posts to the total data
        all_days_data.extend(daily_posts)
        current_date += timedelta(days=1)  # Move to the next day

    # Convert the aggregated posts from all days to a DataFrame
    if all_days_data:
        df = pd.DataFrame(all_days_data)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        return df
    else:
        logger.warning("No posts collected during the given date range.")
        return pd.DataFrame(columns=['text', 'created_at', 'user', 'upvotes', 'num_comments', 'subreddit'])

