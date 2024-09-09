import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def plot_stock_prediction(features, model, symbol):
    data = features[symbol]
    
    # Prepare the data
    X = data.drop(['Target'], axis=1)
    y = data['Target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate confidence intervals
    def prediction_interval(y_true, y_pred, confidence=0.95):
        mse = mean_squared_error(y_true, y_pred)
        n = len(y_true)
        dof = n - 2
        t_value = abs(stats.t.ppf((1 - confidence) / 2, dof))
        sigma = np.sqrt(mse * (1 + 1/n))
        return t_value * sigma

    train_interval = prediction_interval(y_train, y_train_pred)
    test_interval = prediction_interval(y_test, y_test_pred)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Date': X.index,
        'Actual': y,
        'Predicted_Train': pd.Series(y_train_pred, index=X_train.index),
        'Predicted_Test': pd.Series(y_test_pred, index=X_test.index)
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data['Date'], plot_data['Actual'], label='Actual', color='black', alpha=0.6)
    plt.plot(X_train.index, y_train_pred, label='Train Predictions', color='blue', alpha=0.6)
    plt.plot(X_test.index, y_test_pred, label='Test Predictions', color='red', alpha=0.6)
    
    # Add confidence intervals
    plt.fill_between(X_train.index, 
                     y_train_pred - train_interval, 
                     y_train_pred + train_interval, 
                     color='blue', alpha=0.2)
    plt.fill_between(X_test.index, 
                     y_test_pred - test_interval, 
                     y_test_pred + test_interval, 
                     color='red', alpha=0.2)
    
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # This is a placeholder to show how the function would be used
    from model_development.stock_prediction_model import train_and_evaluate
    from feature_engineering.feature_engineer import engineer_features
    import yfinance as yf
    
    # Get some sample data
    stock_data = {'AAPL': yf.download('AAPL', start='2020-01-01', end='2023-01-01')}
    sentiment_scores = pd.Series(np.random.randn(len(stock_data['AAPL'])), index=stock_data['AAPL'].index)
    
    features = engineer_features(sentiment_scores, stock_data)
    model_results = train_and_evaluate(features)
    
    plot_stock_prediction(features, model_results['AAPL']['model'], 'AAPL')