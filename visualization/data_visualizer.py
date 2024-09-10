import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats

def plot_stock_prediction(features, model, symbol, n_splits=5):
    data = features[symbol]
    
    # Prepare the data
    X = data.drop(['Target'], axis=1)
    y = data['Target']
    
    # Make predictions on the entire dataset
    y_pred = model.predict(X)
    
    # Calculate confidence interval
    def prediction_interval(y_true, y_pred, confidence=0.95):
        mse = mean_squared_error(y_true, y_pred)
        n = len(y_true)
        dof = n - 2
        t_value = abs(stats.t.ppf((1 - confidence) / 2, dof))
        sigma = np.sqrt(mse * (1 + 1/n))
        return t_value * sigma

    interval = prediction_interval(y, y_pred)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Date': X.index,
        'Actual': y,
        'Predicted': y_pred
    })
    plot_data.set_index('Date', inplace=True)
    plot_data.sort_index(inplace=True)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data.index, plot_data['Actual'], label='Actual', color='black', alpha=0.6, linewidth=2)
    plt.plot(plot_data.index, plot_data['Predicted'], label='Predicted', color='blue', alpha=0.6, linewidth=2)
    
    # Add confidence interval
    plt.fill_between(plot_data.index, 
                     plot_data['Predicted'] - interval, 
                     plot_data['Predicted'] + interval, 
                     color='blue', alpha=0.2, label='Confidence Interval')
    
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print some metrics
    mse = mean_squared_error(y, y_pred)
    print(f"Overall MSE: {mse:.4f}")

