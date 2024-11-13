from dtuimldmtools import mcnemar
from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold
from Classification_ANN import ClassificationAnn
from Classification_Baseline import ClassificationBaseline
import tabulate
import warnings



seed = 20
K = 10 # Number of folds for cross-validation

CV = KFold(n_splits=K, shuffle=True, random_state=seed)

dataset = Dataset(original_data=False)
attrbute_to_predict = 11
X = dataset.X_mean_std
X = np.delete(X, attrbute_to_predict, 1)
y = dataset.y



results = []
k = 0
for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # ##########################################
    #          Baseline vs Classification      #
    # ##########################################
 
    # Compute the Jeffreys interval
    with  warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod1 = ClassificationAnn(X_train, y_train, h= 10, max_iter=2000, n_replicates=3)
        mod2 = ClassificationBaseline(X_train, y_train)

        yhat = np.empty((X_test.shape[0], 2))
        yhat[:, 0] = mod1.predict(X_test)
        yhat[:, 1] = mod2.predict(X_test)
        alpha = 0.05
        [thetahat, CI, p] = mcnemar(y_test, yhat[:, 0], yhat[:, 1], alpha=alpha)

        results.append((float(thetahat), (float(CI[0]), float(CI[1])), float(p)))
        k += 1


table = tabulate.tabulate(
    results,
    headers = ["theta hat", "ci", "p-value"]
)
print(table)
