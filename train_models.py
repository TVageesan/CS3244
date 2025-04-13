import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from models.model_trainer import ModelTrainer
from models.model_implementations import (
    get_linear_regression_model,
    get_ridge_model,
    get_lasso_model,
    get_random_forest_model,
    get_knn_model,
    get_feature_sets
)

def run_full_comparison(data_path, feature_set='all_features', optimize=False, output_dir='output/model_results'):
    """
    Run a full comparison of all models with the specified feature set.
    
    Args:
        data_path (str): Path to the preprocessed data
        feature_set (str): Name of feature set configuration to use
        optimize (bool): Whether to use hyperparameter optimization
        output_dir (str): Directory to save results
        
    Returns:
        pandas.DataFrame: Comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model trainer
    trainer = ModelTrainer(data_path=data_path)
    
    # Get feature configuration
    feature_sets = get_feature_sets()
    if feature_set not in feature_sets:
        raise ValueError(f"Feature set '{feature_set}' not found. Available sets: {list(feature_sets.keys())}")
    
    feature_config = feature_sets[feature_set]
    print(f"Using feature set: {feature_set} - {feature_config['description']}")
    
    # Load data with feature configuration
    feature_subset = feature_config.get('subset_features', None)
    drop_features = feature_config.get('drop_features', None)
    trainer.load_data(feature_subset=feature_subset, drop_features=drop_features)
    
    # Register models
    trainer.register_model('Linear Regression', get_linear_regression_model())
    
    if optimize:
        trainer.register_model('Ridge (optimized)', get_ridge_model(alpha=None))
        trainer.register_model('Lasso (optimized)', get_lasso_model(alpha=None))
        trainer.register_model('Random Forest (optimized)', get_random_forest_model(optimize=True))
        trainer.register_model('KNN (optimized)', get_knn_model(optimize=True))
    else:
        trainer.register_model('Ridge', get_ridge_model(alpha=1.0))
        trainer.register_model('Lasso', get_lasso_model(alpha=1.0))
        trainer.register_model('Random Forest', get_random_forest_model(n_estimators=100))
        trainer.register_model('KNN', get_knn_model(n_neighbors=5))
    
    # Train and evaluate all models
    for model_name in trainer.models.keys():
        print(f"\nEvaluating {model_name}...")
        trainer.evaluate_model(model_name)
    
    # Compare models
    comparison = trainer.compare_models(metric='r2', ascending=False)
    print("\nModel Comparison (sorted by R²):")
    print(comparison[['r2', 'rmse', 'mae', 'train_time']])
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trainer.save_results(f"{feature_set}_comparison_{timestamp}.json")
    
    # Plot residuals for the best model
    best_model = comparison.index[0]
    print(f"\nBest model: {best_model}")
    fig = trainer.plot_residuals(best_model)
    fig.savefig(os.path.join(output_dir, f"{feature_set}_{best_model.replace(' ', '_')}_residuals_{timestamp}.png"))
    
    return comparison

def evaluate_specific_model(data_path, model_type, feature_set='all_features', optimize=False, output_dir='output/model_results'):
    """
    Evaluate a specific model type.
    
    Args:
        data_path (str): Path to the preprocessed data
        model_type (str): Type of model to evaluate
        feature_set (str): Name of feature set configuration to use
        optimize (bool): Whether to use hyperparameter optimization
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model trainer
    trainer = ModelTrainer(data_path=data_path)
    
    # Get feature configuration
    feature_sets = get_feature_sets()
    if feature_set not in feature_sets:
        raise ValueError(f"Feature set '{feature_set}' not found. Available sets: {list(feature_sets.keys())}")
    
    feature_config = feature_sets[feature_set]
    print(f"Using feature set: {feature_set} - {feature_config['description']}")
    
    # Load data with feature configuration
    feature_subset = feature_config.get('subset_features', None)
    drop_features = feature_config.get('drop_features', None)
    trainer.load_data(feature_subset=feature_subset, drop_features=drop_features)
    
    # Register the requested model
    model_mapping = {
        'linear': ('Linear Regression', get_linear_regression_model()),
        'ridge': ('Ridge', get_ridge_model(alpha=None if optimize else 1.0)),
        'lasso': ('Lasso', get_lasso_model(alpha=None if optimize else 1.0)),
        'rf': ('Random Forest', get_random_forest_model(optimize=optimize)),
        'knn': ('KNN', get_knn_model(optimize=optimize))
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type '{model_type}' not found. Available types: {list(model_mapping.keys())}")
    
    model_name, model = model_mapping[model_type]
    if optimize:
        model_name += " (optimized)"
    
    trainer.register_model(model_name, model)
    
    # Train and evaluate the model
    trainer.evaluate_model(model_name)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trainer.save_results(f"{model_type}_{feature_set}_{timestamp}.json")
    
    # Plot residuals
    fig = trainer.plot_residuals(model_name)
    fig.savefig(os.path.join(output_dir, f"{model_type}_{feature_set}_residuals_{timestamp}.png"))

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate models for property price prediction')
    
    # Main operation mode
    parser.add_argument('mode', choices=['compare', 'single'], 
                        help='Operation mode: compare all models or evaluate a single model')
    
    # Model type (for single mode)
    parser.add_argument('--model', choices=['linear', 'ridge', 'lasso', 'rf', 'knn'],
                        help='Model type for single evaluation')
    
    # Feature set configuration
    parser.add_argument('--features', choices=list(get_feature_sets().keys()), default='all_features',
                        help='Feature set configuration to use')
    
    # Data path
    parser.add_argument('--data', default='output/encoded_data.csv',
                        help='Path to the preprocessed data')
    
    # Optimization flag
    parser.add_argument('--optimize', action='store_true',
                        help='Use hyperparameter optimization')
    
    # Output directory
    parser.add_argument('--output', default='output/model_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        run_full_comparison(args.data, args.features, args.optimize, args.output)
    elif args.mode == 'single':
        if not args.model:
            parser.error("--model is required when mode is 'single'")
        evaluate_specific_model(args.data, args.model, args.features, args.optimize, args.output)

if __name__ == '__main__':
    main()