

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



class RegressionBaseline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Compute the largest class on the training data (most frequent class)

    def predict(self, X_test):
        y_train_mean = np.mean(self.y)
        y_pred = np.full(X_test.shape[0], y_train_mean)
        return y_pred



if __name__ == "__main__":
    dataset = Dataset(original_data=False)

    attribute_to_predict = 3

    y = dataset.X_mean_std[:, attribute_to_predict]

    print(y.mean())
    X = np.zeros_like(y)
    X = X.reshape(-1, 1)
    K = 10
    CV = KFold(n_splits=K, shuffle=True, random_state=20) 

    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        y_train_mean = np.mean(y_train)
        y_pred = np.full(y_test.shape, y_train_mean)
        mse = mean_squared_error(y_test, y_pred)
        print(mse) 

    y_mean_array = np.ones(len(y)) * np.mean(y)
    print("MSE predicted at one:", mean_squared_error(y, y_mean_array))

    model = lambda: np.ones(len(y)) * np.mean(y)

    # def generate_regression_baseline_model(X_train, y_train):
    #     model = lambda x: np.ones(len(x)) * np.mean(y_train)
    #     return model



