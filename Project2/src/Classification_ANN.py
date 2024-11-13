# exercise 8.3.1 
import numpy as np
import sklearn.linear_model as lm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from matplotlib.pyplot import figure, show, title, legend
from scipy.io import loadmat
from scipy import stats
import tabulate

import torch

from dtuimldmtools import dbplotf, train_neural_net, visualize_decision_boundary

from Dataset import Dataset

MAX_ITER = 2000
N_REPLICATES = 3

class ClassificationAnn:
    def __init__(self, X, y, h = 10, max_iter = 10000, n_replicates = 3):
        self.X = X
        self.y = y
        self.h = h
        self.max_iter = max_iter
        self.n_replicates = n_replicates

        C = 7
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], h),  # M features to H hiden units
                torch.nn.ReLU(),  # 1st transfer function
                torch.nn.Linear(h, C),  # C logits
                torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
            )

    def predict(self, X_test): 
        net, _, _ = train_neural_net(
            self.model,
            self.loss_fn,
            X=torch.tensor(self.X, dtype=torch.float),
            y=torch.tensor(self.y, dtype=torch.long),
            n_replicates=self.n_replicates,
            max_iter=self.max_iter,
        )

        softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
        y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
        return y_test_est
        # Determine errors
    


    
def cross_validate(X, y, hidden_units, K):
    CV = KFold(n_splits=K, shuffle=True, random_state=20)
    N, M = X.shape
    test_error = np.empty((K, len(hidden_units)))
    f = 0
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for i, n_hidden_units in enumerate(hidden_units):
            mod = ClassificationAnn(X_train, y_train, n_hidden_units, max_iter=MAX_ITER, n_replicates=N_REPLICATES) 
            y_test_est = mod.predict(X_test)
            # Determine errors
            e = y_test_est != y_test
            test_error[f, i] = np.sum(e) / y_test.shape[0]

        f += 1
        optimal_h_err = np.min(np.mean(test_error, axis=0))
        optimal_h = hidden_units[np.argmin(np.mean(test_error, axis=0))]

        return (optimal_h_err, optimal_h)

if __name__ == "__main__":
    dataset = Dataset(original_data=False)

    # Normalize data
    # X = stats.zscore(X)
    # Load Matlab data file and extract variables of interest
    attrbute_to_predict = 11
    X = dataset.X_mean_std
    X = np.delete(X, attrbute_to_predict, 1)
    y = dataset.y

    # X = X - np.ones((X.shape[0], 1)) * np.mean(X, 0)

    attributeNames = dataset.attributeNames
    classNames = dataset.classNames

    N, M = X.shape
    C = len(classNames)

    # Model fitting and prediction

    # ## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
    # do_pca_preprocessing = False
    # if do_pca_preprocessing:
    #     Y = stats.zscore(X, 0)
    #     U, S, V = np.linalg.svd(Y, full_matrices=False)
    #     V = V.T # pca compontents
    #     # Components to be included as features
    #     k_pca = 2
    #     X = X @ V[:, :k_pca]
    #     N, M = X.shape



    loss_fn = torch.nn.CrossEntropyLoss()

    model = lambda h: torch.nn.Sequential(
            torch.nn.Linear(M, h),  # M features to H hiden units
            torch.nn.ReLU(),  # 1st transfer function
            torch.nn.Linear(h, C),  # C logits
            torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
        )


    K = 10
    CV = KFold(n_splits=K, shuffle=True, random_state=20) 
    hidden_units = range(1, 11)
    k = 0
    error_data = []

    # Training loop
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        print("inner cross validaiton")
        (optimal_h_err, optimal_h) = cross_validate(X_train, y_train, hidden_units, K)
        mod = ClassificationAnn(X_train, y_train, optimal_h, max_iter=MAX_ITER, n_replicates=N_REPLICATES)
        y_test_est = mod.predict(X_test)
        e = y_test_est != y_test
        e = np.sum(e) / y_test.shape[0]
        error_data.append([optimal_h, e])
        k += 1

    table = tabulate.tabulate(
        error_data,
        headers = ["Optimal amout of hidden units", "Test error"]
    )
    print(table)

