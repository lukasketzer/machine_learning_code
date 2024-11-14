# exercise 8.2.6
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from Dataset import Dataset
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from dtuimldmtools import draw_neural_net, train_neural_net
import tabulate

# Parameters for neural network classifier
MAX_ITER = 5_000
N_REPLICATES = 3  # number of networks trained in each k-fold


class RegressionAnn:
    def __init__(self, X, y, h=10, max_iter=10000, n_replicates=3):
        self.X = X
        self.y = y
        self.h = h
        self.max_iter = max_iter
        self.n_replicates = n_replicates

        # Define the model
        self.loss_fn = (
            torch.nn.MSELoss()
        )  # notice how this is now a mean-squared-error loss
        self.model = lambda: torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.h),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(self.h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )

    def predict(self, X_test):
        net, _, _ = train_neural_net(
            self.model,
            self.loss_fn,
            X=torch.tensor(self.X, dtype=torch.float),
            y=torch.tensor(self.y, dtype=torch.float),
            n_replicates=self.n_replicates,
            max_iter=self.max_iter,
        )
        y_test_est = net(X_test)
        return y_test_est


def cross_validate(model, loss_fn, X, y, hidden_units, K):
    CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=20)
    N, M = X.shape
    test_error = np.empty((K, len(hidden_units)))
    f = 0
    for train_index, test_index in CV.split(X, y):
        # print("Cross Validation has finished %i of %i; relative: %i", f, K, f/K)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for i, n_hidden_units in enumerate(hidden_units):
            mod = RegressionAnn(
                X_train,
                y_train,
                h=n_hidden_units,
                n_replicates=N_REPLICATES,
                max_iter=MAX_ITER,
            )

            y_test_est = mod.predict(X_test)
            # Determine errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
            test_error[f, i] = mse[0]

        f += 1
        optimal_h_err = np.min(test_error)
        optimal_h = hidden_units[np.argmin(np.mean(test_error, axis=0))]

        return (optimal_h_err, optimal_h)


if __name__ == "__main__":
    # Load Matlab data file and extract variables of interest
    data_set = Dataset(original_data=False)
    mat_data = data_set.X_mean_std
    # Normalize data
    mat_data = stats.zscore(mat_data)
    attributeNames = [np.str_(name) for name in data_set.attributeNames]
    print(attributeNames)

    parameter_index = 3
    y = mat_data[:, [parameter_index]]  # weight parameter
    X = mat_data[
        :, np.arange(len(mat_data[0])) != parameter_index
    ]  # source: https://stackoverflow.com/questions/19286657/index-all-except-one-item-in-python
    # X = (mat_data[:, :parameter_index].T + mat_data[:, (parameter_index+1):].T).T  # the rest of features

    N, M = X.shape

    ## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
    do_pca_preprocessing = False
    if do_pca_preprocessing:
        Y = stats.zscore(X, 0)
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        V = V.T
        # Components to be included as features
        k_pca = 3
        X = X @ V[:, :k_pca]
        N, M = X.shape

    # K-fold crossvalidation
    K = 10  #  # only three folds to speed up this example
    N_hidden_units_count = 20
    CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=20)
    n_hidden_units_range = range(
        1, N_hidden_units_count + 1
    )  # 20 hidden layers for Regression ANN as 9 produced good results
    error_data = []

    # Define the model
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
    model = lambda n_hidden_units: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )

    # Training Model
    # print("Training model of type:\n\n{}\n".format(str(model())))
    errors = []  # make a list for storing generalizaition error in each loop
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])

        # print("inner cross validaiton")
        (optimal_h_err, optimal_h) = cross_validate(
            model, loss_fn, X_train, y_train, n_hidden_units_range, K
        )

        mod = RegressionAnn(
            X_train, y_train, h=optimal_h, n_replicates=N_REPLICATES, max_iter=MAX_ITER
        )

        # Determine estimated class labels for test set
        y_test_est = mod.predict(X_test)

        # Determine errors and errors
        # se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        # mse1 = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        mse = mean_squared_error(y_test_est.data.numpy(), y_test.data.numpy())
        errors.append([optimal_h, mse])  # store error rate for current CV fold

    # def generate_regression_ANN_model(X_train, y_train):
    #     net, _, _ = train_neural_net(
    #         mod,
    #         loss_fn,
    #         X=torch.Tensor(X_train),
    #         y=torch.Tensor(y_train),
    #         n_replicates=N_REPLICATES,
    #         max_iter=MAX_ITER,
    #     )
    #     return lambda x: np.ndarray.flatten(net(torch.Tensor(x)).detach().numpy())

    table = tabulate.tabulate(
        errors, headers=["Optimal amout of hidden units", "Test error"]
    )
    print(table)
