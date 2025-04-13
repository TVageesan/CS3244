import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_knn(n = 5):
    model = KNeighborsRegressor(n_neighbors=n)
    return model

def test_knn(X_train, X_test, y_train, n = 5):
    model = get_knn(n)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def test_k_values(train_X, train_y):
    """
    Chart the r^2 score against k-value for the model
    Find the optimal "maximum" point for k
    
    Args:
        df: DataFrame with features and target column 'resale_price'
        
    Returns:
        int: Optimal k value
    """
    k_lim = len(train_X)**0.5
    
    # Define k value ranges
    k_values = [i for i in range(1, round(k_lim+1), 2)]
    
    # Test broad range of k values
    rsq_values = []
    general_k_values = []
    
    for i in range(0, len(k_values), 10):
        knn_reg = KNeighborsRegressor(n_neighbors=k_values[i])
        rsq_value = cross_val_score(knn_reg, train_X, train_y, cv=5, scoring="r2")
        rsq_values.append(np.mean(rsq_value))
        general_k_values.append(k_values[i])
        print(f'{k_values[i]} nearest neighbour: R^2 value: {np.mean(rsq_value)}')
    
    # Test detailed range for zoomed view
    zoomed_rsq_values = []
    zoomed_k_values = list(range(1, 42, 2))
    
    for i in zoomed_k_values:
        knn_reg = KNeighborsRegressor(n_neighbors=i)
        rsq_value = cross_val_score(knn_reg, train_X, train_y, cv=5, scoring="r2")
        zoomed_rsq_values.append(np.mean(rsq_value))
        print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value)}')
    
    # Find best k value
    best_k_idx = np.argmax(zoomed_rsq_values)
    best_k = zoomed_k_values[best_k_idx]
    best_r2 = zoomed_rsq_values[best_k_idx]
    
    # Plot broader range
    plt.figure(figsize=(10, 6))
    plt.plot(general_k_values, rsq_values, color="red", marker="o", label=f"k: 1 to {general_k_values[-1]}")
    plt.title("Cross Validation Mean R^2 vs k")
    plt.grid(True)
    plt.xlabel("Number of Nearest Neighbours (increment of 10)")
    plt.ylabel("Mean R^2")
    plt.legend()
    plt.show()
    
    # Plot zoomed view
    plt.figure(figsize=(10, 6))
    plt.plot(zoomed_k_values, zoomed_rsq_values, color="blue", marker="o", label="k: 1 to 41")
    
    # Annotate the best k value
    plt.annotate(f'({best_k}, {round(best_r2, 3)})',
                xy=(best_k, best_r2),
                xytext=(best_k+3, best_r2+0.0001),
                arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA=0.8, shrinkB=0.005, lw=3.2, mutation_scale=5),
                fontsize=10)
    
    plt.grid(True)
    plt.title("Cross Validation Mean R^2 vs k")
    plt.xlabel("Number of Nearest Neighbours (increment of 2)")
    plt.ylabel("Mean R^2")
    plt.legend()
    plt.show()
    
    # Final evaluation with best k
    print(f"Best k value is {best_k} with R^2 score of {round(best_r2, 3)}")
    
    return best_k

