import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from models.knn import get_knn
from models.linear_regression import get_linear
from models.random_forest import get_rf
from models.regression_tree import get_rt
from models.validate import prepare_split, evaluate_model
from preprocessing.encode import encode_data
from utils.file_utils import read_csv

def run_model_comparison(feature_configs, data_path='output/clean_data.csv'):
    """
    Compare multiple models across different feature configurations.
    
    Args:
        feature_configs (list): List of dictionaries with feature configurations
        data_path (str): Path to the cleaned data CSV
        
    Returns:
        dict: Results of all models across all configurations
    """
    # Read the base data
    clean_data = read_csv(data_path)
    
    # Dictionary to store results by configuration
    results = {}
    
    # Setup model factories
    model_factories = {
        'KNN': get_knn,
        'Linear': lambda: get_linear('linear'),
        'Ridge': lambda: get_linear('ridge'),
        'Lasso': lambda: get_linear('lasso'),
        'RandomForest': get_rf,
        'RegressionTree': get_rt
    }
    
    # For each feature configuration
    for config_idx, config in enumerate(feature_configs):
        config_name = config.get('name', f'Config {config_idx}')
        print(f"\n{'='*50}")
        print(f"Running with configuration: {config_name}")
        print(f"{'='*50}")
        
        # Apply encoding with this configuration
        encoded_data = encode_data(
            clean_data,
            encoding_method=config.get('encoding_method', 'one_hot'),
            handle_outliers=config.get('handle_outliers', False),
            moving_window=config.get('moving_window', False),
            cyclic_month=config.get('cyclic_month', False),
            normal_year=config.get('normal_year', False),
            normal_price=config.get('normal_price', False),
            spatial_features=config.get('spatial_features', False)
        )
        
        # Split the data
        train_X, test_X, train_y, test_y = prepare_split(encoded_data)
        
        # Initialize results for this configuration
        config_results = {}
        
        # Train and evaluate each model
        for model_name, model_factory in model_factories.items():
            print(f"\nTraining {model_name}...")
            
            # Create model instance
            model = model_factory()
            
            # Train model
            model.fit(train_X, train_y)
            
            # Evaluate
            eval_results = evaluate_model(model, test_X, test_y)
            
            print(f"{model_name} - RMSE: {eval_results['rmse']:.2f}, MAE: {eval_results['mae']:.2f}, R²: {eval_results['r2']:.4f}")
            
            # Store results
            config_results[model_name] = {
                'rmse': eval_results['rmse'],
                'mae': eval_results['mae'],
                'r2': eval_results['r2'],
                'predictions': eval_results['predictions']
            }
        
        results[config_name] = config_results
    
    return results

def plot_model_comparison(results, metric='r2'):
    """
    Plot the performance of all models across different configurations.
    
    Args:
        results (dict): Results from run_model_comparison
        metric (str): Metric to compare ('r2', 'rmse', or 'mae')
    """
    # Convert results to DataFrame for easier plotting
    data = []
    
    for config_name, config_results in results.items():
        for model_name, model_metrics in config_results.items():
            data.append({
                'Configuration': config_name,
                'Model': model_name,
                'R²': model_metrics['r2'],
                'RMSE': model_metrics['rmse'],
                'MAE': model_metrics['mae']
            })
    
    df = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    if metric == 'r2':
        title = 'R² Score by Model and Configuration'
        ylabel = 'R² Score'
        metric_col = 'R²'
    elif metric == 'rmse':
        title = 'RMSE by Model and Configuration'
        ylabel = 'RMSE'
        metric_col = 'RMSE'
    else:  # 'mae'
        title = 'MAE by Model and Configuration'
        ylabel = 'MAE'
        metric_col = 'MAE'
    
    # Plot
    ax = sns.barplot(x='Configuration', y=metric_col, hue='Model', data=df)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature Configuration', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on the bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + (0.01 if metric == 'r2' else 5000),
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', rotation=90, fontsize=8
        )
    
    plt.tight_layout()
    plt.show()


def summarize_best_models(results):
    """
    Summarize the best model for each configuration based on R² score.
    
    Args:
        results (dict): Results from run_model_comparison
        
    Returns:
        pandas.DataFrame: Summary of best models
    """
    summary = []
    
    for config_name, config_results in results.items():
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, metrics in config_results.items():
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = model_name
        
        summary.append({
            'Configuration': config_name,
            'Best Model': best_model,
            'R²': best_r2,
            'RMSE': config_results[best_model]['rmse'],
            'MAE': config_results[best_model]['mae']
        })
    
    summary_df = pd.DataFrame(summary).sort_values('R²', ascending=False)
    return summary_df

def get_results_df(results):
    """
    Convert results dictionary to a DataFrame for easier analysis.
    
    Args:
        results (dict): Results from run_model_comparison
        
    Returns:
        pandas.DataFrame: DataFrame with all metrics
    """
    # Convert to DataFrame for metrics across all configs
    all_metrics = []
    for config_name, models in results.items():
        for model_name, metrics in models.items():
            row = {
                'Configuration': config_name,
                'Model': model_name,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            }
            all_metrics.append(row)
    
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df

