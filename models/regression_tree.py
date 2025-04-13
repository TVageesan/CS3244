import numpy as np
import pandas as pd
from sklearn.model_selection import  cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def get_rt():
    model = DecisionTreeRegressor(max_depth = 16, random_state=42)
    return model

def test_tree_depth(X_train, y_train):
    r2_scores = []
    depths = list(range(3,21))
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=23)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"Depth {depth}: Mean R² = {scores.mean()}")
        r2_scores.append(scores.mean())
        
    plt.figure(figsize=(10, 6))
    plt.plot(depths, r2_scores, marker='o', linestyle='-', color='red')
    plt.title("Cross validated mean R² vs regression tree depth")
    plt.xlabel("Regression tree depth")
    plt.ylabel("Mean R²")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_rt(X_train, X_test, y_train, y_test):
    # train a regression tree startingi with a depth of 16

    rt = get_rt()
    rt.fit(X_train, y_train)

    y_pred = rt.predict(X_test)

    print('Decision Tree accuracy for training set: %f' % rt.score(X_train, y_train))
    print('Decision Tree accuracy for test set: %f' % rt.score(X_test, y_test))
    print(f"Mean absolute error: ±${mean_absolute_error(y_test, y_pred)}")
    return y_pred