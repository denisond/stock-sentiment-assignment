from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate(features):
    results = {}
    
    for symbol, data in features.items():
        logger.info(f"Training model for {symbol}")
        
        if data.empty:
            logger.warning(f"No data available for {symbol}. Skipping this symbol.")
            continue
        
        # Prepare the data
        X = data.drop(['Target'], axis=1)
        y = data['Target']
        
        if len(X) < 2:
            logger.warning(f"Insufficient data for {symbol}. At least 2 samples required. Skipping this symbol.")
            continue
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[symbol] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        logger.info(f"Model for {symbol} - MSE: {mse:.4f}, R2: {r2:.4f}")
    
    if not results:
        raise ValueError("No models could be trained due to insufficient data.")
    
    return results

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2022-01-01', end='2023-01-01')
    sample_data = pd.DataFrame({
        'Returns': np.random.randn(len(dates)),
        'Sentiment': np.random.randn(len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Target': np.random.randn(len(dates))
    }, index=dates)
    
    features = {'SAMPLE': sample_data}
    results = train_and_evaluate(features)
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"MSE: {result['mse']:.4f}")
        print(f"R2: {result['r2']:.4f}")
        print("Top 5 important features:")
        for feature, importance in sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{feature}: {importance:.4f}")