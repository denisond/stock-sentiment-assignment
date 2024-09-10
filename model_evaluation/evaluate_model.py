import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate(stock_prediction_results):
    for symbol, results in stock_prediction_results.items():
        print(f"\nStock Prediction Model Evaluation for {symbol}:")
        print(f"Train Mean Squared Error: {results['train_mse']:.4f}")
        print(f"Train R-squared Score: {results['train_r2']:.4f}")
        print(f"Test Mean Squared Error: {results['test_mse']:.4f}")
        print(f"Test R-squared Score: {-results['test_r2']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        feature_importance = results['feature_importance']
        sorted_idx = sorted(feature_importance, key=feature_importance.get, reverse=True)
        pos = range(len(feature_importance))
        plt.bar(pos, [feature_importance[i] for i in sorted_idx])
        plt.xticks(pos, [i for i in sorted_idx], rotation=90)
        plt.title(f'Feature Importance for {symbol}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()
