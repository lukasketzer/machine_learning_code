# exercise 8.2.6
import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from Dataset import Dataset
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import draw_neural_net, train_neural_net
import tabulate

# Parameters for neural network classifier
MAX_ITER = 2000
N_REPLICATES = 3  # number of networks trained in each k-fold


def cross_validate(model, X, y, hidden_units, K):
    CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=20)
    N, M = X.shape
    test_error = np.empty((K, len(hidden_units)))
    f = 0
    for train_index, test_index in CV.split(X, y):
        #print("Cross Validation has finished %i of %i; relative: %i", f, K, f/K)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for i, n_hidden_units in enumerate(hidden_units):
            mod = lambda: model(n_hidden_units)
            net, _, _ = train_neural_net(
                mod,
                loss_fn,
                X=(X_train),
                y=(y_train),
                n_replicates=N_REPLICATES,
                max_iter=MAX_ITER,
            )

            y_test_est = net(X_test)
            # Determine errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean            
            test_error[f, i] = mse[0]

        f += 1
        optimal_h_err = np.min(test_error)
        optimal_h = hidden_units[np.argmin(np.mean(test_error, axis=0))]

        return (optimal_h_err, optimal_h)


# Load Matlab data file and extract variables of interest
data_set = Dataset(original_data=False)
mat_data = data_set.X_mean_std
attributeNames = [np.str_(name) for name in data_set.attributeNames]
print(attributeNames)
parameter_index = 3
y = mat_data[:, [parameter_index]]  # weight parameter
X = mat_data[:, np.arange(len(mat_data[0])) != parameter_index] # source: https://stackoverflow.com/questions/19286657/index-all-except-one-item-in-python
#X = (mat_data[:, :parameter_index].T + mat_data[:, (parameter_index+1):].T).T  # the rest of features

N, M = X.shape
C = 2 # unused at regression

# Normalize data
X = stats.zscore(X)

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
K = 10  # only three folds to speed up this example
CV = model_selection.KFold(n_splits=K, shuffle=True, random_state=20)
n_hidden_units_range = range(1,K)
error_data = []

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]
# Define the model
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
model = lambda n_hidden_units: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
    # no final tranfer function, i.e. "linear output"
)

# Training Model
#print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])
    
    #print("inner cross validaiton")
    (optimal_h_err, optimal_h) = cross_validate(model, X_train, y_train, n_hidden_units_range, K)
    mod = lambda: model(optimal_h)

    # Train the net on training data
    #net, final_loss, learning_curve = train_neural_net(
    net, _, _ = train_neural_net(
        mod,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=N_REPLICATES,
        max_iter=MAX_ITER,
    )

    #print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_test_est = net(X_test)

    # Determine errors and errors
    se = (y_test_est.float() - y_test.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
    errors.append([optimal_h, mse])  # store error rate for current CV fold

    # Display the learning curve for the best net in the current fold
    #(h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
    #h.set_label("CV fold {0}".format(k + 1))
    #summaries_axes[0].set_xlabel("Iterations")
    #summaries_axes[0].set_xlim((0, MAX_ITER))
    #summaries_axes[0].set_ylabel("Loss")
    #summaries_axes[0].set_title("Learning curves")


table = tabulate.tabulate(
    errors,
    headers = ["Optimal amout of hidden units", "Test error"]
)
print(table)

"""
# Display the MSE across folds
summaries_axes[1].bar(
    np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
)
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_xticks(np.arange(1, K + 1))
summaries_axes[1].set_ylabel("MSE")
summaries_axes[1].set_title("Test mean-squared-error")

print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print(
    "\nEstimated generalization error, RMSE: {0}".format(
        round(np.sqrt(np.mean(errors)), 4)
    )
)

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of
# the true/known value - these values should all be along a straight line "y=x",
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10, 10))
y_est = y_test_est.data.numpy()
y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
plt.plot(axis_range, axis_range, "k--")
plt.plot(y_true, y_est, "ob", alpha=0.25)
plt.legend(["Perfect estimation", "Model estimations"])
plt.title("Body Weight: estimated versus true value (for last CV-fold)")
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel("True value")
plt.ylabel("Estimated value")
plt.grid()

plt.show()

print("Ran Exercise 8.2.5")
"""