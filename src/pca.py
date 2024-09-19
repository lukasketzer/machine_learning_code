import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
filename = "./data.csv"

df = pd.read_csv(filename)

raw_data = df.values
cols = range(0, len(raw_data[0]))

"""
Data needs to be cleand up. In the dataset, values like "Sometimes" or "Car" are used.
Those need to be turned into numbers
"""
# problematic_columns = [n for n, i in enumerate(raw_data[0]) if type(i) == str] # list of columns that are not numbers but strings

clean_data = np.empty_like(raw_data)
labels = []
# TODO: code is shit and not useful
for i, n in enumerate(raw_data[0]):
    classLabels = raw_data[:, i]  # -1 takes the last feature
    classNames = np.unique(classLabels)
    classDict = dict(zip(classNames, range(len(classNames))))
    labels.append(classDict)

    if type(n) != str:
        clean_data[:, i] = raw_data[:, i]
    else:
        attributeNames = np.asarray(df.columns[cols])
        y = np.array([classDict[cl] for cl in classLabels]) # assigns numbers to the differen strings / values
        clean_data[:, i] = y


X = clean_data[:, cols]

N, M = X.shape

C = len(classNames)

# subtract the means from the featues
# Data is now centerd to the center of the coordinate system
Y = X - np.ones((N, 1)) * X.mean(axis=0) # axis = 0 is the vertical axis on a coordinate-system
Y = Y.astype(float)

# divide standart deviation
Y2 = Y * (1 / np.std(Y, axis=0, ddof=1))


# svd
U, S, V = svd(Y2, full_matrices=False)

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()


# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
# plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()


