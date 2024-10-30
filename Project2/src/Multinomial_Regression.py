from Dataset import Dataset
import numpy as np
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, show, title
from dtuimldmtools import visualize_decision_boundary

# TODO: Currently using an 80/20 train-test split; should we try k-fold cross-validation bc it might be better?

dataset = Dataset()

# chosen_attributes = ['Age', 'FAF']
chosen_attributes = ['Gender', 'Age', 'Height', 'family_history_with_overweight', 'NCP', 'CH2O', 'FAF', 'TUE', 'CALC', 'MTRANS']

attributeNames = dataset.attributeNames
chosen_attribute_indexes = np.nonzero(np.isin(attributeNames, chosen_attributes))[0]
attributeNames = [attributeNames[i] for i in chosen_attribute_indexes] # keep only the chosen ones

X = dataset.X_mean_std # Input data
y = dataset.y # Corresponding target output

np.random.seed(42) # Set the random seed for reproducibility
num_samples = X.shape[0] # Get the number of samples
shuffled_indices = np.random.permutation(num_samples) # Generate a random permutation of indices

X = X[:, chosen_attribute_indexes] # keep only chosen attributes

# Shuffle both the input data (X) and the target output (y) using the shuffled indices
X = X[shuffled_indices]
y = y[shuffled_indices]

X_train = X[:int(0.8 * dataset.N)]  # Splitting into 80% train
X_test = X[int(0.8 * dataset.N):]   # and 20% test

y_train = y[:int(0.8 * dataset.N)] # y = labels (target output)
y_test = y[int(0.8 * dataset.N):]

classNames = dataset.classNames
C = dataset.C  # Number of classes

N, M = X.shape # matrix dimensions (N rows, M columns)

# Model fitting and prediction

# Multinomial logistic regression
logreg = lm.LogisticRegression(
    solver="lbfgs", multi_class="multinomial", tol=1e-4, random_state=1
)
logreg.fit(X_train, y_train)

# To display coefficients use print(logreg.coef_). For a 4 class problem with a
# feature space, these weights will have shape (4, 2).

# Number of miss-classifications
print(
    "Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}".format(
        np.sum(logreg.predict(X_test) != y_test), len(y_test)
    )
)

predict = lambda x: np.argmax(logreg.predict_proba(x), 1)
figure(2, figsize=(9, 9))
visualize_decision_boundary(
    predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames
)
title("LogReg decision boundaries")

show()

print("Ran Exercise 8.3.2")
