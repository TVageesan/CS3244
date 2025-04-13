import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('encoded_data.csv') 

town_columns = [col for col in df.columns if col.startswith("town_")]
type_columns = [col for col in df.columns if col.startswith("flat_type_")]
model_columns = [col for col in df.columns if col.startswith("flat_model_")]

features = ["floor_area_sqm", "avg_floor", "remaining_lease", "year"] + town_columns + type_columns + model_columns
X = df[features]
y = df["resale_price"]

# split into 20% test / 80% train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Using the mean of 5 fold validation for R² scores, find optimal tree depth

r2_scores = []
depths = list(range(3,21))
for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=23)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Depth {depth}: Mean R² = {scores.mean()}")
    r2_scores.append(scores.mean())

# Plot out the mean of the R² values for better visualisation

plt.figure(figsize=(10, 6))
plt.plot(depths, r2_scores, marker='o', linestyle='-', color='red')
plt.title("Cross validated mean R² vs regression tree depth")
plt.xlabel("Regression tree depth")
plt.ylabel("Mean R²")
plt.grid(True)

plt.tight_layout()
plt.show()

print("We decided to choose a tree depth of 16 through 5-fold validation as not only does the mean R² values begin to plaeaeu, ")
print ("with R² values improvements becoming less than 1% beyond 16. Beyond this, a larger depth means a more complex model, and thus")
print ("the chance that the model might overfit to the training data increases even further.")

# train a regression tree startingi with a depth of 16

rt = DecisionTreeRegressor(max_depth = 16, random_state=23)
rt.fit(X_train, y_train)

y_pred = rt.predict(X_test)

print('Decision Tree accuracy for training set: %f' % rt.score(X_train, y_train))
print('Decision Tree accuracy for test set: %f' % rt.score(X_test, y_test))


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute error: ±${mae}")