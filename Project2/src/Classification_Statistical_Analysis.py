from dtuimldmtools import mcnemar
from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold
from Classification_ANN import ClassificationAnn
from Classification_Baseline import ClassificationBaseline
from Classification_Multinomial_Regression import ClassificationMultinomialRegression
import tabulate
from collections import defaultdict



seed = 20
K = 10 # Number of folds for cross-validation

CV = KFold(n_splits=K, shuffle=True, random_state=seed)

dataset = Dataset(original_data=False)
attrbute_to_predict = 11
X = dataset.X_mean_std
X = np.delete(X, attrbute_to_predict, 1)
y = dataset.y



results = defaultdict(list)
k = 0
alpha = 0.05

for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    mod1 = ClassificationAnn(X_train, y_train, h= 5, max_iter=5000, n_replicates=3)
    mod2 = ClassificationBaseline(X_train, y_train)
    mod3 = ClassificationMultinomialRegression(X_train, y_train, 1e-5)

    # ##########################################
    #          Baseline vs ANN                 #
    # ##########################################
 
    # Compute the Jeffreys interval
    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1.predict(X_test) #Ann
    yhat[:, 1] = mod2.predict(X_test) #Baseline
    [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

    results["BaselineANN"].append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))

    # ##########################################
    #          Baseline vs LogReg              #
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod2.predict(X_test) #baseline
    yhat[:, 1] = mod3.predict(X_test) #logreg
    [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

    results["BaselineLogReg"].append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))

    # ##########################################
    #          ANN vs LogReg                   #
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1.predict(X_test) # ann
    yhat[:, 1] = mod3.predict(X_test) # logreg
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
