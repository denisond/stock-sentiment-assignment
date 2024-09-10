from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate(features, n_splits=5):
    results = {}
    
    for symbol, data in features.items():
        logger.info(f"Training model for {symbol}")
        
        if data.empty:
            logger.warning(f"No data available for {symbol}. Skipping this symbol.")
            continue
        
        # Prepare the data
        X = data.drop(['Target'], axis=1)
        y = data['Target']
        
        if len(X) < n_splits + 1:
            logger.warning(f"Insufficient data for {symbol}. At least {n_splits + 1} samples required. Skipping this symbol.")
            continue
        
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize lists to store results
        train_mse_scores = []
        train_r2_scores = []
        test_mse_scores = []
        test_r2_scores = []
        feature_importances = []
        
        # Perform time series cross-validation
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate the model on train set
            train_mse_scores.append(mean_squared_error(y_train, y_train_pred))
            train_r2_scores.append(r2_score(y_train, y_train_pred))
            
            # Evaluate the model on test set
            test_mse_scores.append(mean_squared_error(y_test, y_test_pred))
            test_r2_scores.append(r2_score(y_test, y_test_pred))
            
            feature_importances.append(model.feature_importances_)
        
        # Calculate average scores and feature importances
        avg_train_mse = np.mean(train_mse_scores)
        avg_train_r2 = np.mean(train_r2_scores)
        avg_test_mse = np.mean(test_mse_scores)
        avg_test_r2 = np.mean(test_r2_scores)
        avg_feature_importance = np.mean(feature_importances, axis=0)
        
        # Train a final model on all data
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        
        results[symbol] = {
            'model': final_model,
            'train_mse': avg_train_mse,
            'train_r2': avg_train_r2,
            'test_mse': avg_test_mse,
            'test_r2': avg_test_r2,
            'feature_importance': dict(zip(X.columns, avg_feature_importance))
        }
        
        logger.info(f"Model for {symbol}:")
        logger.info(f"  Average Train MSE: {avg_train_mse:.4f}, Average Train R2: {avg_train_r2:.4f}")
        logger.info(f"  Average Test MSE: {avg_test_mse:.4f}, Average Test R2: {-avg_test_r2:.4f}")
    
    if not results:
        raise ValueError("No models could be trained due to insufficient data.")
    
    return results
