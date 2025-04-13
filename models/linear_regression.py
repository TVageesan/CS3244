import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../output/encoded_data.csv')

X = df.drop(columns=['resale_price'])
y = df['resale_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Print results
print(f"RMSE: {rmse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"R2: {r2:.5f}")
print(f"Cross-validation R2 scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean():.5f}")


#find best alpha params

import warnings
warnings.simplefilter('ignore')

param_grid = {'alpha': [0.1, 0.5, 1.0, 10, 100]}
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_alpha_l = grid_search.best_params_['alpha']
print(f"Best alpha for Lasso: {best_alpha_l}")

grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_alpha_r = grid_search.best_params_['alpha']
print(f"Best alpha for Ridge: {best_alpha_r}")



lasso = Lasso(alpha=best_alpha_l)
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


ridge = Ridge(alpha=best_alpha_r)
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


# Plot of Actal vs Predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Resale Price')
plt.ylabel('Predicted Resale Price')
plt.title('Actual vs Predicted Resale Price')
plt.grid(True)
plt.show()


# Residuals plot
residuals = y_test - y_pred_test

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=50)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show() # Residuals should be normally distributed


