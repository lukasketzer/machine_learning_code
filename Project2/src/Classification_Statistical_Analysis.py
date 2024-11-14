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



results = {
    "BaselineANN":[[],[],[]],
    "BaselineLogReg": [[], [],[]],
    "LogRegANN": [[], [], []]
}
k = 0
alpha = 0.05

def stack(name, y_true, yhat1, yhat2):
    results[name][0] += list(y_true)
    results[name][1] += list(yhat1)
    results[name][2] += list(yhat2)
    # results[name] = np.vstack(
    #     (
    #         results[name],
    #         np.array((y_true, yhat1, yhat2))
    #     )
    # )

for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    mod1 = ClassificationAnn(X_train, y_train, h= 5, max_iter=100, n_replicates=3)
    mod2 = ClassificationBaseline(X_train, y_train)
    mod3 = ClassificationMultinomialRegression(X_train, y_train, 1e-5)

    # ##########################################
    #          Baseline vs ANN                 #
    # ##########################################
 
    # Compute the Jeffreys interval
    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1.predict(X_test) #Ann
    yhat[:, 1] = mod2.predict(X_test) #Baseline
    stack("BaselineANN", y_test, yhat[:, 0], yhat[:, 1])
    # ##########################################
    #          Baseline vs LogReg              #
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod2.predict(X_test) #baseline
    yhat[:, 1] = mod3.predict(X_test) #logreg
    stack("BaselineLogReg", y_test, yhat[:, 0], yhat[:, 1])



    # ##########################################
    #          ANN vs LogReg                   #
    # ##########################################
 
    # Compute the Jeffreys interval

    yhat = np.empty((X_test.shape[0], 2))
    yhat[:, 0] = mod1.predict(X_test) # ann
    yhat[:, 1] = mod3.predict(X_test) # logreg
    stack("LogRegANN", y_test, yhat[:, 0], yhat[:, 1])

    k += 1

stringstore = ["BaselineANN", "BaselineLogReg", "LogRegANN"]
print(results)

with open("output.txt", "w") as file:
    for i in stringstore:
        print(i)
        file.write(i)
        [thetahat, CI, p] = mcnemar(np.array(results[i][0]), np.array(results[i][1]), np.array(results[i][2]), alpha=alpha)
        file.write(str([thetahat, CI, p]))

