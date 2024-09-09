from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def train_and_evaluate(data, text_column='content', sentiment_threshold=0.1):
    # Prepare the data
    X = data[text_column]
    y = data['sentiment'].apply(lambda x: 'positive' if x > sentiment_threshold else 'negative' if x < -sentiment_threshold else 'neutral')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_vectorized)
    report = classification_report(y_test, y_pred)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'classification_report': report
    }

if __name__ == "__main__":
    from data_collection.news_collector import collect_news
    from data_preprocessing.sentiment_analyzer import analyze_sentiment
    
    news_data = collect_news()
    news_data['sentiment'] = analyze_sentiment(news_data['content'])
    
    results = train_and_evaluate(news_data)
    print(results['classification_report'])
