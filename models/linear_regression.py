import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_linear(model_type='linear', alpha = 0.5):
    if model_type.lower() == 'ridge':
        return Ridge(alpha=alpha, random_state=42)
    elif model_type.lower() == 'lasso':
        return Lasso(alpha=alpha, random_state=42)
    else:
        return LinearRegression()
    
def test_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Print results
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"R2: {r2:.5f}")
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2: {cv_scores.mean():.5f}")    
    return y_pred_test

def test_ridge(X_train, X_test, y_train, y_test, alpha = 0.5):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # Ridge Model Evaluation
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mse_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # Print results for Ridge
    print(f"Ridge RMSE: {rmse_ridge:.5f}")
    print(f"Ridge MAE: {mae_ridge:.5f}")
    print(f"Ridge R2: {r2_ridge:.5f}")
    return y_pred_ridge

def test_lasso(X_train, X_test, y_train, y_test, alpha = 0.5):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    # Lasso Model Evaluation
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    rmse_lasso = np.sqrt(mse_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Print results for Lasso
    print(f"Lasso RMSE: {rmse_lasso:.5f}")
    print(f"Lasso MAE: {mae_lasso:.5f}")
    print(f"Lasso R2: {r2_lasso:.5f}")
    return y_pred_lasso

def test_alpha_value(train_X, train_y):
    import warnings
    warnings.simplefilter('ignore')

    param_grid = {'alpha': [0.1, 0.5, 1.0, 10, 100]}
    grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_X, train_y)
    best_alpha_l = grid_search.best_params_['alpha']
    print(f"Best alpha for Lasso: {best_alpha_l}")

    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_X, train_y)
    best_alpha_r = grid_search.best_params_['alpha']
    print(f"Best alpha for Ridge: {best_alpha_r}")
