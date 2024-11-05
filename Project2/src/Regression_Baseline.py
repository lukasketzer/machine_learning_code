################### Baseline Model for Project 2

import csv
import numpy as np
from Dataset import Dataset
# read file

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = Dataset()

attribute_to_predict = 3

y = dataset.X_raw[:, attribute_to_predict]
X = np.zeros_like(y)
X = X.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)


