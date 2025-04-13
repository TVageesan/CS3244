from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

def get_linear_regression_model():
    """Get a basic linear regression model"""
    return LinearRegression()

def get_ridge_model(alpha=1.0, cv=5):
    """
    Get a Ridge regression model with optional optimization.
    
    Args:
        alpha (float or None): Regularization strength. If None, use GridSearchCV
        cv (int): Cross-validation folds for optimization
        
    Returns:
        Ridge: The configured model
    """
    if alpha is None:
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        return GridSearchCV(Ridge(), param_grid, cv=cv, scoring='neg_mean_squared_error')
    else:
        return Ridge(alpha=alpha)

def get_lasso_model(alpha=1.0, cv=5):
    """
    Get a Lasso regression model with optional optimization.
    
    Args:
        alpha (float or None): Regularization strength. If None, use GridSearchCV
        cv (int): Cross-validation folds for optimization
        
    Returns:
        Lasso: The configured model
    """
    if alpha is None:
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        return GridSearchCV(Lasso(), param_grid, cv=cv, scoring='neg_mean_squared_error')
    else:
        return Lasso(alpha=alpha)

def get_random_forest_model(n_estimators=100, max_depth=None, cv=5, optimize=False):
    """
    Get a Random Forest Regressor model with optional optimization.
    
    Args:
        n_estimators (int): Number of trees
        max_depth (int or None): Maximum tree depth
        cv (int): Cross-validation folds for optimization
        optimize (bool): Whether to use GridSearchCV
        
    Returns:
        RandomForestRegressor: The configured model
    """
    if optimize:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }
        return GridSearchCV(
            RandomForestRegressor(random_state=42), 
            param_grid, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
    else:
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

def get_knn_model(n_neighbors=5, weights='uniform', optimize=False, cv=5):
    """
    Get a K-Nearest Neighbors regressor with optional optimization.
    
    Args:
        n_neighbors (int): Number of neighbors
        weights (str): Weight function ('uniform' or 'distance')
        optimize (bool): Whether to use GridSearchCV
        cv (int): Cross-validation folds for optimization
        
    Returns:
        KNeighborsRegressor: The configured model
    """
    if optimize:
        # Find optimal k value through grid search
        param_grid = {
            'n_neighbors': list(range(1, 21, 2)),  # Test odd values from 1 to 19
            'weights': ['uniform', 'distance']
        }
        return GridSearchCV(
            KNeighborsRegressor(), 
            param_grid, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
    else:
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

def get_feature_sets():
    """
    Define different feature sets for testing.
    
    Returns:
        dict: Dictionary of feature set configurations
    """
    return {
        'all_features': {
            'description': 'All available features',
            'drop_features': []
        },
        'no_town': {
            'description': 'Without town categorical features (using coordinates instead)',
            'drop_features': [col for col in pd.read_csv('output/encoded_data.csv').columns 
                             if col.startswith('town_')]
        },
        'no_coordinates': {
            'description': 'Without longitude and latitude (using town instead)',
            'drop_features': ['longitude', 'latitude']
        },
        'minimal': {
            'description': 'Only essential features',
            'subset_features': ['floor_area_sqm', 'avg_floor', 'remaining_lease', 
                               'distance_to_nearest_mrt', 'year', 'month_sin', 'month_cos']
        }
    }