import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import mplcursors

df = pd.read_csv("KNN_data.csv")


#preparation of data
y = df["resale_price"] #label

town_columns = [col for col in df.columns if col.startswith("town_")]
town_columns

X = df.drop(columns = "resale_price") #features
#print(X)
X = X.drop(columns = town_columns)
# X = X.drop(columns = ["latitude", "longitude"])

#Split data set into train and test set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 32)
# train_X1, test_X1, train_y1, test_y1 = train_test_split(X_without_town, y, test_size = 0.2, random_state = 32)
# train_X2, test_X2, train_y2, test_y2 = train_test_split(X_without_coords, y, test_size = 0.2, random_state = 32)

# Implementing k Nearest Neighbour Algorithm
# Using OneHotEncoded Town features
k_lim = len(train_X)**0.5 #finding max k value for kNN
#print(k_lim)

rsq_values = []
k_values = [i for i in range(1,round(k_lim+1),2)]
#print(k_values)

for i in range(0,len(k_values), 10):
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors = k_values[i])
    rsq_value = cross_val_score(knn_reg, train_X, train_y, cv = 5, scoring = "r2")
    rsq_values.append(np.mean(rsq_value))
    print(f'{k_values[i]} nearest neighbour: R^2 value: {np.mean(rsq_value)}')

zoomed_rsq_values = []
for i in range(1,42,2):
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors = i)
    rsq_value = cross_val_score(knn_reg, train_X, train_y, cv = 5, scoring = "r2")
    zoomed_rsq_values.append(np.mean(rsq_value))
    print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value)}')

# Finding best k value
general_k_values = []
for i in range(0,len(k_values), 10):
    general_k_values.append(k_values[i])


# plotting R^2 values against k_values
plt.plot(general_k_values, rsq_values, color = "red", marker = "o", label = "k: 1 to 481")
plt.title("Cross Validation Mean R^2 v.s k")
plt.grid(True)
plt.xlabel("Number of Nearest Neighbours (increment of 10)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()


# zoom into maximum region
plt.plot(k_values[0:21], zoomed_rsq_values, color = "blue", marker = "o", label = "k: 1 to 41")
#plt.text(k_values[2], zoomed_rsq_values[2], f'({k_values[2]}, {round(zoomed_rsq_values[2],3)})', fontsize=12, ha='left', va='bottom')
plt.annotate(f'({k_values[3]}, {round(zoomed_rsq_values[3],3)})',
             xy=(k_values[3], zoomed_rsq_values[3]), 
             xytext=(k_values[3]+3, zoomed_rsq_values[3]+0.0001),  # where the label appears
             arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA = 0.8, shrinkB=0.005, lw = 3.2, mutation_scale = 5),
             fontsize=10)
plt.grid(True)
mplcursors.cursor(hover=True)
plt.title("Cross Validation Mean R^2 v.s k")
plt.xlabel("Number of Nearest Neighbours (increment of 2)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()

# Testing 5-Nearest Neighbour model
knn_reg_5  = neighbors.KNeighborsRegressor(n_neighbors = 5)

knn_reg_5.fit(train_X, train_y)
pred_y = knn_reg_5.predict(test_X)

# Checking fit of kNN
mae = mean_absolute_error(test_y, pred_y)
print(f'Mean absolute error between actual and predicted resale price: {round(mae,2)}')

mse = mean_squared_error(test_y, pred_y)
print(f'Mean squared error between actual and predicted resale price: {round(mse,2)}')