# exercise 7.3.1
import scipy.stats as st
import sklearn.linear_model
import sklearn.tree

import numpy as np
from scipy import stats
from Dataset import Dataset
from sklearn import model_selection

from dtuimldmtools import *
from dtuimldmtools.statistics.statistics import correlated_ttest

from Regression_ANN import generate_regression_ANN_model
from Regression_Baseline import generate_regression_baseline_model

loss = 2
# Load Matlab data file and extract variables of interest
data_set = Dataset(original_data=False)
mat_data = data_set.X_mean_std
# Normalize data
mat_data = stats.zscore(mat_data)
attributeNames = [np.str_(name) for name in data_set.attributeNames]
print(attributeNames)

parameter_index = 3
y = mat_data[:, [parameter_index]]  # weight parameter
X = mat_data[:, np.arange(len(mat_data[0])) != parameter_index] # source: https://stackoverflow.com/questions/19286657/index-all-except-one-item-in-python
#X = (mat_data[:, :parameter_index].T + mat_data[:, (parameter_index+1):].T).T  # the rest of features# This script crates predictions from three KNN classifiers using cross-validation

K = 10 # We presently set J=K
J = K
m = 1
r = []
kf = model_selection.KFold(n_splits=K)

for dm in range(m):
    y_true = []
    yhat = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        mA = generate_regression_ANN_model(X_train, y_train)
        mB = generate_regression_baseline_model(X_train, y_train)  # baseline model currently based on complete data

        yhatA = mA(X_test)
        yhatB = mB(X_test)
        y_true.append(y_test)
        print(yhatA)
        #print(yhatB)
        yhat.append( np.concatenate([yhatA, yhatB], axis=0) )

        r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)

#print("p setup is: ", p_setupII)
#print("CI is", CI_setupII)


if m == 1:
    y_true = np.concatenate(y_true)[:,0]
    yhat = np.concatenate(yhat)

    # note our usual setup I ttest only makes sense if m=1.
    zA = np.abs(y_true - yhat[:,0] ) ** loss
    zB = np.abs(y_true - yhat[:,1] ) ** loss
    z = zA - zB

    CI_setupI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_setupI = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

    print("p  setup 2 and 1", p_setupII, p_setupI )
    print("CI setup 2 and 1", CI_setupII, CI_setupI )
