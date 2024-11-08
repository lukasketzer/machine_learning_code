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

MAX_ITER = 10000
N_REPLICATES = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

def MSE(y_test, y_pred):
    return np.square(y_test - y_pred).sum(axis=0)

def cross_validate(model, X, y, hidden_units, K):
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
            mod = lambda: model(n_hidden_units)
            net, _, _ = train_neural_net(
                mod,
                loss_fn,
                X=torch.tensor(X_train, dtype=torch.float).to(device=device),
                y=torch.tensor(y_train, dtype=torch.long).to(device=device),
                n_replicates=N_REPLICATES,
                max_iter=MAX_ITER,
            )

            softmax_logits = net(torch.tensor(X_test, dtype=torch.float).to(device=device))
            y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            # Determine errors
            e = y_test_est != y_test
            test_error[f, i] = np.sum(e) / y_test.shape[0]

        f += 1
        optimal_h_err = np.min(np.mean(test_error, axis=0))
        optimal_h = hidden_units[np.argmin(np.mean(test_error, axis=0))]

        return (optimal_h_err, optimal_h)


dataset = Dataset(original_data=False)


# Load Matlab data file and extract variables of interest
X = dataset.X_mean_std
y = dataset.y
# X = X - np.ones((X.shape[0], 1)) * np.mean(X, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


attributeNames = dataset.attributeNames
classNames = dataset.classNames

N, M = X.shape
C = len(classNames)
# Model fitting and prediction

# ## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X, 0)
    U, S, V = np.linalg.svd(Y, full_matrices=False)
    V = V.T # pca compontents
    # Components to be included as features
    k_pca = 2
    X = X @ V[:, :k_pca]
    N, M = X.shape



loss_fn = torch.nn.CrossEntropyLoss()

model = lambda h: torch.nn.Sequential(
        torch.nn.Linear(M, h),  # M features to H hiden units
        torch.nn.ReLU(),  # 1st transfer function
        torch.nn.Linear(h, C),  # C logits
        torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
    ).to(device=device)

K = 10
CV = KFold(n_splits=K, shuffle=True, random_state=20) 
hidden_units = range(1, 10)
k = 0
error_data = []
for train_index, test_index in CV.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    print("inner cross validaiton")
    (optimal_h_err, optimal_h) = cross_validate(model, X_train, y_train, hidden_units, K)
    mod = lambda: model(optimal_h)
    net, _, _ = train_neural_net(
        mod,
        loss_fn,
        X=torch.tensor(X_train, dtype=torch.float).to(device=device),
        y=torch.tensor(y_train, dtype=torch.long).to(device=device),
        n_replicates=N_REPLICATES,
        max_iter=MAX_ITER,
    )

    softmax_logits = net(torch.tensor(X_test, dtype=torch.float).to(device=device))
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    e = y_test_est != y_test
    e = np.sum(e) / y_train.shape[0]
    error_data.append([optimal_h, e])
    k += 1

table = tabulate.tabulate(
    error_data,
    headers = ["Optimal amout of hidden units", "Test error"]
)
print(table)

