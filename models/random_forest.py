import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_rf():
    model = RandomForestRegressor(random_state=42)
    return model
    
def test_rf(X_train, X_test, y_train, y_test):
    model = get_rf()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error:', mse)
    print('R^2 Score:', r2)
    return y_pred