# %%
from Dataset import Dataset
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
import sklearn.feature_selection
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    plot,
    loglog,
    semilogx,
    show,
    subplot,
    savefig,
    title,
    xlabel,
    ylabel,
)
from scipy import stats

from dtuimldmtools import bmplot, feature_selector_lr, rlr_validate



class RegressionRegularization:
    def __init__(self, X, y, l = 10):
        # add ones to the front 
        self.X = X
        self.y = y
        self.l = l

        # normalize and standardize
        self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), 1)
        self.M = self.X.shape[1]
        self.mu = np.mean(self.X[:, 1:], 0)
        self.sigma = np.std(self.X[:, 1:], 0)
        self.sigma[self.sigma == 0] = 1e-14
        self.X[:, 1:] = (self.X[:, 1:] - self.mu) / self.sigma


    def predict(self, X_test):

        X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
        X_test[:, 1:] = (X_test[:, 1:] - self.mu) / self.sigma

        lambdaI = self.l * np.eye(self.M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        Xty = self.X.T @ self.y
        XtX = self.X.T @ self.X
        w = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        y_test_pred = X_test @ w
        return y_test_pred

