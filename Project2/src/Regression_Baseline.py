

import csv
import numpy as np
from Dataset import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
# read file

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def MSE(y_test, y_pred):
    return np.square(y_test - y_pred).sum(axis=0) / y_test.shape[0]

dataset = Dataset()

attribute_to_predict = 3


y = dataset.X_mean_std[:, attribute_to_predict]
X = np.zeros_like(y)
X = X.reshape(-1, 1)
print(f"Mean: {np.mean(y)}")
K = 10
CV = KFold(n_splits=K, shuffle=True, random_state=20) 

for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    

    y_train_mean = np.mean(y_train)
    y_pred = np.full(y_test.shape, y_train_mean)
    mse = mean_squared_error(y_test, y_pred) * y_test.shape[0]
    print(mse) 






