from Dataset import Dataset
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = Dataset()
    y = dataset.X_mean_std[:, 3] # we want to predcit the weight
    X = np.delete(dataset.X_mean_std, 3, 1) # delete fourth column out of matrix
    # X = dataset.X_mean_std[:, 2] # height i think

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)    

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    

    # Predict alcohol content
    residual = y_pred - y_test

    # Display scatter plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(y_test, y_pred, ".")
    plt.xlabel("Standardized Weight (true)")
    plt.ylabel("Standardized Weight (estimated)")
    plt.subplot(2, 1, 2)
    plt.hist(residual, 40)

    plt.show()