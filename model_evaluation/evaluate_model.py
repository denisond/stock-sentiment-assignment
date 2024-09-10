import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(sentiment_model_results, stock_prediction_results):
    # Evaluate sentiment model
    print("Sentiment Model Evaluation:")
    print(sentiment_model_results['classification_report'])
    
    # Evaluate stock prediction model
    print("\nStock Prediction Model Evaluation:")
    for symbol, results in stock_prediction_results.items():
        print(f"\n{symbol}:")
        print(f"Mean Squared Error: {results['mse']:.4f}")
        print(f"R-squared Score: {results['r2']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
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
