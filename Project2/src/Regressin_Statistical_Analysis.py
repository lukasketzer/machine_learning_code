# exercise 7.3.1
import scipy.stats as st
import sklearn.tree

from scipy import stats
from dtuimldmtools import mcnemar
from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold
import tabulate
from collections import defaultdict

from Regression_ANN import generate_regression_ANN_model
from Regression_Baseline import generate_regression_baseline_model
from Regression_Regularisation import generate_regression_regular_model

loss = 2
# Load data file and extract variables of interest
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

seed = 20
K = 20 # Number of folds for cross-validation

CV = KFold(n_splits=K, shuffle=True, random_state=seed)


results = defaultdict(list)
k = 0
alpha = 0.05

for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    mod1 = generate_regression_ANN_model(X_train, y_train)
    mod2 = generate_regression_baseline_model(X_train, y_train)
    mod3 = generate_regression_regular_model(X_train, y_train)
    """
    mod1 = ClassificationAnn(X_train, y_train, h= 5, max_iter=5000, n_replicates=3)
    mod2 = ClassificationBaseline(X_train, y_train)
    mod3 = ClassificationMultinomialRegression(X_train, y_train, 1e-5)
    """

    # ##########################################
    #          Baseline vs ANN      #
    # ##########################################
 
    # Compute the Jeffreys interval
    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1(X_test) #Ann
    yhat[:, 1] = mod2(X_test) #Baseline
    [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

    results["BaselineANN"].append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))

    # ##########################################
    #          Baseline vs LogReg      
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod2(X_test) #baseline
    yhat[:, 1] = mod3(X_test) #logreg
    [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

    results["BaselineLogReg"].append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))

    # ##########################################
    #          ANN vs LogReg      
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1(X_test) # ann
    yhat[:, 1] = mod3(X_test) # logreg
    [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

    results["LogRegANN"].append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))
    k += 1

with open("output.txt", "w") as file:
    for i in results:
        print(i)
        file.write(i)
        table = tabulate.tabulate(
            results[i],
            headers = ["theta hat", "ci", "p-value"]
        )
        file.write(table)
        file.write("\n")
        print(table)
