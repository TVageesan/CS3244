import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

def get_knn(n = 6):
    model = KNeighborsRegressor(n_neighbors=n)
    return model

def test_knn(X_train, X_test, y_train, n = 6):
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
    # Implementing k Nearest Neighbour Algorithm
    k_lim = len(train_X)**0.5 #finding max k value for kNN
    #print(k_lim)

    k_values = [i for i in range(1,round(k_lim+1))]

    rsq_values = []
    general_k_values = []

    for i in range(0,len(k_values), 20):
        knn_reg = get_knn(k_values[i])
        rsq_value = cross_val_score(knn_reg, train_X, train_y, cv = 5, scoring = "r2")
        rsq_values.append(np.mean(rsq_value))
        general_k_values.append(k_values[i])
        print(f'{k_values[i]} nearest neighbour: R^2 value: {np.mean(rsq_value)}')

    zoomed_rsq_values = []
    zoomed_k_values = list(range(1, 42))

    for i in zoomed_k_values:
        knn_reg = get_knn(i)
        rsq_value = cross_val_score(knn_reg, train_X, train_y, cv = 5, scoring = "r2")
        zoomed_rsq_values.append(np.mean(rsq_value))
        print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value)}')
    
    general_k_values = []
    for i in range(0,len(k_values), 20):
        general_k_values.append(k_values[i])


    # plotting R^2 values against k_values
    plt.plot(general_k_values, rsq_values, color = "red", marker = "o", label = "K: 1 to 481")
    plt.title("Cross Validation Mean R² v.s K")
    plt.grid(True)
    plt.xlabel("Number of Nearest Neighbours (K) (increment of 20)")
    plt.ylabel("Mean R²")
    plt.legend()
    plt.show()
    
    
    best_rsq_index = np.argmax(zoomed_rsq_values)
    best_rsq = zoomed_rsq_values[best_rsq_index]
    best_k = k_values[best_rsq_index]
    zoomed_k_values = list(range(1,42))
    # zoom into maximum region
    plt.plot(zoomed_k_values, zoomed_rsq_values, color = "blue", marker = "o", label = "K: 1 to 41")
    #plt.text(k_values[2], zoomed_rsq_values[2], f'({k_values[2]}, {round(zoomed_rsq_values[2],3)})', fontsize=12, ha='left', va='bottom')
    plt.annotate(f'({best_k}, {round(best_rsq,3)})',
                xy=(best_k, best_rsq), 
                xytext=(best_k+3, best_rsq+0.0001),  # where the label appears
                arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA = 0.8, shrinkB=0.005, lw = 3.2, mutation_scale = 5),
                fontsize=10)
    plt.grid(True)
    mplcursors.cursor(hover=True)
    plt.title("Cross Validation Mean R² v.s K")
    plt.xlabel("Number of Nearest Neighbours (K)")
    plt.ylabel("Mean R²")
    plt.legend()
    plt.show()
