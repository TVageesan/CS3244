import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_split(df, test_size = 0.2, random_state = 32):
    y = df["resale_price"] 
    X = df.drop(columns = "resale_price") 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_X, test_X, train_y, test_y


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on the test set
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def cross_validate(model, X, y, cv=5, scoring='r2'):
    """
    Perform cross-validation on a model
    
    Args:
        model: Model to validate
        X: Features
        y: Target values
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric to use
        
    Returns:
        tuple: (mean_score, std_score)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()
