# exercise 7.3.1
from scipy import stats
from dtuimldmtools.statistics.statistics import correlated_ttest
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
    
    mod1 = generate_regression_ANN_model(X_train, y_train)
    mod2 = generate_regression_baseline_model(X_train, y_train)
    mod3 = generate_regression_regular_model(X_train, y_train)
    """
    mod1 = ClassificationAnn(X_train, y_train, h= 5, max_iter=5000, n_replicates=3)
    mod2 = ClassificationBaseline(X_train, y_train)
    mod3 = ClassificationMultinomialRegression(X_train, y_train, 1e-5)
    """
    
    yhatA = mod1(X_test) #Ann
    yhatB = mod2(X_test) #Baseline
    yhatR = mod3(X_test) #reg

    r[0].append(np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ))
    r[1].append(np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatR-y_test) ** loss ))
    r[2].append(np.mean( np.abs( yhatB-y_test ) ** loss - np.abs( yhatR-y_test) ** loss ))
    
    k += 1
    
# Compute the Correlated Ttest
stringstore = ["ANN-Baseline", "ANN-Regression", "Baseline-Regression"]
for i in range(3):
    p_setupII, CI_setupII = correlated_ttest(r[0], rho, alpha=alpha)
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
