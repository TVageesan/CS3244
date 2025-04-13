import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

class ModelTrainer:
    """
    A unified interface for training, evaluating, and comparing different models.
    """
    def __init__(self, data_path='output/encoded_data.csv', test_size=0.2, random_state=42):
        """
        Initialize the model trainer with data loading and splitting functionality.
        
        Args:
            data_path (str): Path to the preprocessed dataset
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Create results directory if it doesn't exist
        os.makedirs('output/model_results', exist_ok=True)
        
    def load_data(self, feature_subset=None, drop_features=None):
        """
        Load and split the data for training and testing.
        
        Args:
            feature_subset (list, optional): List of features to use (if None, use all)
            drop_features (list, optional): List of features to exclude
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Separate target variable
        y = df['resale_price']
        X = df.drop(columns=['resale_price'])
        
        # Apply feature selection if specified
        if feature_subset:
            X = X[feature_subset]
            print(f"Using feature subset: {feature_subset}")
        
        if drop_features:
            X = X.drop(columns=[f for f in drop_features if f in X.columns])
            print(f"Dropped features: {drop_features}")
            
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Data split: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def register_model(self, name, model, params=None):
        """
        Register a model for training and evaluation.
        
        Args:
            name (str): Name identifier for the model
            model: The model object
            params (dict, optional): Parameters for the model
        """
        self.models[name] = {
            'model': model,
            'params': params or {},
            'fitted': False
        }
        print(f"Registered model: {name}")
        
    def train_model(self, name):
        """
        Train a registered model.
        
        Args:
            name (str): Name of the model to train
            
        Returns:
            object: The trained model
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
            
        if self.X_train is None:
            self.load_data()
            
        model_info = self.models[name]
        model = model_info['model']
        
        print(f"Training model: {name}")
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        self.models[name]['fitted'] = True
        self.models[name]['train_time'] = train_time
        
        print(f"Model {name} trained in {train_time:.2f} seconds")
        return model
    
    def evaluate_model(self, name, cv=5):
        """
        Evaluate a trained model and store metrics.
        
        Args:
            name (str): Name of the model to evaluate
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Evaluation metrics
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not registered")
            
        model_info = self.models[name]
        if not model_info['fitted']:
            self.train_model(name)
            
        model = model_info['model']
        
        # Make predictions on test set
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, 
                                   pd.concat([self.X_train, self.X_test]), 
                                   pd.concat([self.y_train, self.y_test]), 
                                   cv=cv, scoring='r2')
        
        # Store results
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std()),
            'train_time': model_info['train_time']
        }
        
        self.results[name] = {
            'metrics': metrics,
            'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
            'actuals': self.y_test.tolist()
        }
        
        print(f"Model {name} evaluation:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  CV R² (mean): {cv_scores.mean():.4f}")
        
        return metrics
    
    def compare_models(self, metric='r2', ascending=False):
        """
        Compare all evaluated models based on a specified metric.
        
        Args:
            metric (str): Metric to use for comparison
            ascending (bool): Sort order
            
        Returns:
            pandas.DataFrame: Comparison table
        """
        if not self.results:
            print("No models have been evaluated yet")
            return None
            
        comparison = {}
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison[name] = metrics
            
        df_comparison = pd.DataFrame(comparison).T
        df_sorted = df_comparison.sort_values(by=metric, ascending=ascending)
        
        return df_sorted
    
    def plot_residuals(self, name):
        """
        Plot residuals for a trained model.
        
        Args:
            name (str): Name of the model
            
        Returns:
            matplotlib.figure.Figure: The residual plot
        """
        if name not in self.results:
            raise ValueError(f"Model '{name}' has not been evaluated")
            
        result = self.results[name]
        y_pred = np.array(result['predictions'])
        y_true = np.array(result['actuals'])
        
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        ax[0].scatter(y_true, y_pred, alpha=0.5)
        ax[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title(f'{name}: Actual vs Predicted')
        
        # Residuals
        ax[1].hist(residuals, bins=50, alpha=0.7)
        ax[1].axvline(0, color='r', linestyle='--')
        ax[1].set_xlabel('Residuals')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filename=None):
        """
        Save evaluation results to a JSON file.
        
        Args:
            filename (str, optional): Custom filename, defaults to timestamp
            
        Returns:
            str: Path to the saved file
        """
        if not self.results:
            print("No results to save")
            return None
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
            
        filepath = os.path.join('output/model_results', filename)
        
        # Extract serializable results (exclude large prediction arrays)
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                'metrics': result['metrics'],
                # Optionally include sample predictions for verification
                'sample_predictions': result['predictions'][:5] if len(result['predictions']) > 5 else result['predictions'],
                'sample_actuals': result['actuals'][:5] if len(result['actuals']) > 5 else result['actuals']
            }
            
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        return filepath