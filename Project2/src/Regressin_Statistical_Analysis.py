# exercise 7.3.1
from scipy import stats
from dtuimldmtools.statistics.statistics import correlated_ttest
from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold
import tabulate
import torch
from collections import defaultdict

from Regression_ANN import RegressionAnn
from Regression_Baseline import RegressionBaseline
from Regression_Regularisation import RegressionRegularization

# Parameters for neural network classifier
MAX_ITER = 5000
N_REPLICATES = 3  # number of networks trained in each k-fold



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
rho = 1/K
loss = 2
#y_true = []
r = [[],[],[]]
#p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)

for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    # X_train = torch.Tensor(X[train_index, :])
    # y_train = torch.Tensor(y[train_index])
    # X_test = torch.Tensor(X[test_index, :])
    # y_test = torch.Tensor(y[test_index])

    mod1 = RegressionAnn(torch.Tensor(X_train), torch.Tensor(y_train), n_replicates=N_REPLICATES, max_iter=MAX_ITER, h = 5)
    mod2 = RegressionBaseline(X_train, y_train)
    mod3 = RegressionRegularization(X_train, y_train, l = 5)
    """
    mod1 = ClassificationAnn(X_train, y_train, h= 5, max_iter=5000, n_replicates=3)
    mod2 = ClassificationBaseline(X_train, y_train)
    mod3 = ClassificationMultinomialRegression(X_train, y_train, 1e-5)
    """
    
    yhatA = mod1.predict(torch.Tensor(X_test)).data.numpy() #Ann
    yhatB = mod2.predict(X_test) #Baseline
    yhatR = mod3.predict(X_test) #reg

    r[0].append(np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ))
    r[1].append(np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatR-y_test) ** loss ))
    r[2].append(np.mean( np.abs( yhatB-y_test ) ** loss - np.abs( yhatR-y_test) ** loss ))
    
    k += 1
    
# Compute the Correlated Ttest
stringstore = ["ANN-Baseline", "ANN-Regression", "Baseline-Regression"]
for i in range(3):
    p_setupII, CI_setupII = correlated_ttest(r[i], rho, alpha=alpha)
    results[stringstore[i]].append((float(CI_setupII[0]), float(CI_setupII[1]), float(p_setupII)))

with open("output_Regression.txt", "w") as file:
    for i in results:
        print(i)
        file.write(i)
        table = tabulate.tabulate(
            results[i],
            headers = ["ci", "p-value"]
        )
        file.write(table)
        file.write("\n")
        print(table)
