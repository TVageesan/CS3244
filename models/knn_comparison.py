import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import mplcursors

import pandas as pd
import math

def add_nearest_mrt(geocoded_df):
    mrt_df = pd.read_csv('mrt.csv')

    # Haversine distance function (in kilometers)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    # Pre-compute MRT coordinates
    mrt_coords = list(zip(mrt_df['lat'], mrt_df['lng']))

    # Compute nearest MRT distance per row
    def compute_nearest_mrt(lat, lon):
        return min(haversine(lat, lon, mrt_lat, mrt_lon) for mrt_lat, mrt_lon in mrt_coords)

    geocoded_df['distance_to_nearest_mrt'] = geocoded_df.apply(
        lambda row: compute_nearest_mrt(row['latitude'], row['longitude']),
        axis=1
    )

    return geocoded_df


df_geocode = pd.read_csv("geocoded.csv")
df_mrt = add_nearest_mrt(df_geocode)

df_mrt.to_csv("geocode_with_mrt.csv", index = False)

df_source = pd.read_csv('geocode_with_mrt.csv') 
df_target = pd.read_csv("encoded_data.csv")

#combine dataset to use single data file (include distance to nearest mrt, longitude and latitude (last 2 will replace categorical feature: town))
columns_to_copy = ["distance_to_nearest_mrt", "latitude", "longitude"]
df_copy = df_source[columns_to_copy]

df_combined = pd.concat([df_target, df_copy], axis = 1)

df_combined.to_csv("KNN_data.csv", index = False)

df_combined = pd.read_csv("KNN_data.csv")

#preparation of data
y = df_combined["resale_price"] #label

town_columns = [col for col in df_combined.columns if col.startswith("town_")]
town_columns

X = df_combined.drop(columns = "resale_price") #features
#print(X)
X_without_town = X.drop(columns = town_columns)
X_without_coords = X.drop(columns = ["latitude", "longitude"])

#Split data set into train and test set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 32)
train_X1, test_X1, train_y1, test_y1 = train_test_split(X_without_town, y, test_size = 0.2, random_state = 32)
train_X2, test_X2, train_y2, test_y2 = train_test_split(X_without_coords, y, test_size = 0.2, random_state = 32)

# Implementing k Nearest Neighbour Algorithm

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


# Implementing k Nearest Neighbour Algorithm
# Using Geographical Coordinates

k_lim = len(train_X)**0.5 #finding max k value for kNN
#print(k_lim)

rsq_values1 = []
k_values = [i for i in range(1,round(k_lim+1),2)]
#print(k_values)

for i in range(0,len(k_values), 10):
    knn_reg1 = neighbors.KNeighborsRegressor(n_neighbors = k_values[i])
    rsq_value1 = cross_val_score(knn_reg1, train_X1, train_y1, cv = 5, scoring = "r2")
    rsq_values1.append(np.mean(rsq_value1))
    print(f'{k_values[i]} nearest neighbour: R^2 value: {np.mean(rsq_value1)}')

# Implementing k Nearest Neighbour Algorithm
# Using OneHotEncoded Town features

k_lim = len(train_X)**0.5 #finding max k value for kNN
#print(k_lim)

rsq_values2 = []
k_values = [i for i in range(1,round(k_lim+1),2)]
#print(k_values)

for i in range(0,len(k_values), 10):
    knn_reg2 = neighbors.KNeighborsRegressor(n_neighbors = k_values[i])
    rsq_value2 = cross_val_score(knn_reg2, train_X2, train_y2, cv = 5, scoring = "r2")
    rsq_values2.append(np.mean(rsq_value2))
    print(f'{k_values[i]} nearest neighbour: R^2 value: {np.mean(rsq_value2)}')

zoomed_rsq_values = []
for i in range(1,42,2):
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors = i)
    rsq_value = cross_val_score(knn_reg, train_X, train_y, cv = 5, scoring = "r2")
    zoomed_rsq_values.append(np.mean(rsq_value))
    print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value)}')


zoomed_rsq_values1 = []
for i in range(1,42,2):
    knn_reg1 = neighbors.KNeighborsRegressor(n_neighbors = i)
    rsq_value1 = cross_val_score(knn_reg1, train_X1, train_y1, cv = 5, scoring = "r2")
    zoomed_rsq_values1.append(np.mean(rsq_value1))
    print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value1)}')

zoomed_rsq_values2 = []
for i in range(1,42,2):
    knn_reg2 = neighbors.KNeighborsRegressor(n_neighbors = i)
    rsq_value2 = cross_val_score(knn_reg2, train_X2, train_y2, cv = 5, scoring = "r2")
    zoomed_rsq_values2.append(np.mean(rsq_value2))
    print(f'{i} nearest neighbour: R^2 value: {np.mean(rsq_value2)}')

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

# Finding best k value
general_k_values = []
for i in range(0,len(k_values), 10):
    general_k_values.append(k_values[i])


# plotting R^2 values against k_values
plt.plot(general_k_values, rsq_values1, color = "red", marker = "o", label = "k: 1 to 481")
plt.title("Cross Validation Mean R^2 v.s k")
plt.grid(True)
plt.xlabel("Number of Nearest Neighbours (increment of 10)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()


# Finding best k value
general_k_values = []
for i in range(0,len(k_values), 10):
    general_k_values.append(k_values[i])


# plotting R^2 values against k_values
plt.plot(general_k_values, rsq_values2, color = "red", marker = "o", label = "k: 1 to 481")
plt.title("Cross Validation Mean R^2 v.s k")
plt.grid(True)
plt.xlabel("Number of Nearest Neighbours (increment of 10)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()

# zoom into maximum region
plt.plot(k_values[0:21], zoomed_rsq_values, color = "blue", marker = "o", label = "k: 1 to 41")
#plt.text(k_values[2], zoomed_rsq_values[2], f'({k_values[2]}, {round(zoomed_rsq_values[2],3)})', fontsize=12, ha='left', va='bottom')
plt.annotate(f'({k_values[2]}, {round(zoomed_rsq_values[2],3)})',
             xy=(k_values[2], zoomed_rsq_values[2]), 
             xytext=(k_values[2]+3, zoomed_rsq_values[2]+0.0001),  # where the label appears
             arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA = 0.8, shrinkB=0.005, lw = 3.2, mutation_scale = 5),
             fontsize=10)
plt.grid(True)
mplcursors.cursor(hover=True)
plt.title("Cross Validation Mean R^2 v.s k")
plt.xlabel("Number of Nearest Neighbours (increment of 2)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()

plt.cla()
# zoom into maximum region
plt.plot(k_values[0:21], zoomed_rsq_values1, color = "blue", marker = "o", label = "k: 1 to 41")
#plt.text(k_values[3], zoomed_rsq_values1[3], f'({k_values[3]}, {round(zoomed_rsq_values1[3],3)})', fontsize=12, ha='left', va='bottom')
plt.annotate(f'({k_values[3]}, {round(zoomed_rsq_values1[3],3)})',
             xy=(k_values[3], zoomed_rsq_values1[3]), 
             xytext=(k_values[3]+3, zoomed_rsq_values1[3]+0.0001),  # where the label appears
             arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA = 0.8, shrinkB=0.005, lw = 3.2, mutation_scale = 5),
             fontsize=10)
plt.grid(True)
mplcursors.cursor(hover=True)
plt.title("Cross Validation Mean R^2 v.s k")
plt.xlabel("Number of Nearest Neighbours (increment of 2)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()


# zoom into maximum region
plt.plot(k_values[0:21], zoomed_rsq_values2, color = "blue", marker = "o", label = "k: 1 to 41")
#plt.text(k_values[2], zoomed_rsq_values2[2], f'({k_values[2]}, {round(zoomed_rsq_values2[2],3)})', fontsize=12, ha='left', va='bottom')
plt.annotate(f'({k_values[2]}, {round(zoomed_rsq_values2[2],3)})',
             xy=(k_values[2], zoomed_rsq_values2[2]), 
             xytext=(k_values[2]+3, zoomed_rsq_values2[2]+0.0001),  # where the label appears
             arrowprops=dict(arrowstyle='->', facecolor='black', shrinkA = 0.8, shrinkB=0.005, lw = 3.2, mutation_scale = 5),
             fontsize=10)
plt.grid(True)
mplcursors.cursor(hover=True)
plt.title("Cross Validation Mean R^2 v.s k")
plt.xlabel("Number of Nearest Neighbours (increment of 2)")
plt.ylabel("Mean R^2")
plt.legend()
plt.show()

knn_reg_5  = neighbors.KNeighborsRegressor(n_neighbors = 5)

knn_reg_5.fit(train_X, train_y)
pred_y = knn_reg_5.predict(test_X)
knn_reg_5.fit(train_X1, train_y1)
pred_y1 = knn_reg_5.predict(test_X1)
knn_reg_5.fit(train_X2, train_y2)
pred_y2 = knn_reg_5.predict(test_X2)

# Checking fit of kNN
mae = mean_absolute_error(test_y, pred_y)
print(f'Mean absolute error between actual and predicted resale price: {round(mae,2)}')

mse = mean_squared_error(test_y, pred_y)
print(f'Mean squared error between actual and predicted resale price: {round(mse,2)}')

# Checking fit of kNN
mae = mean_absolute_error(test_y, pred_y1)
print(f'Mean absolute error between actual and predicted resale price: {round(mae,2)}')

mse = mean_squared_error(test_y, pred_y1)
print(f'Mean squared error between actual and predicted resale price: {round(mse,2)}')

# Checking fit of kNN
mae = mean_absolute_error(test_y, pred_y2)
print(f'Mean absolute error between actual and predicted resale price: {round(mae,2)}')

mse = mean_squared_error(test_y, pred_y2)
print(f'Mean squared error between actual and predicted resale price: {round(mse,2)}')