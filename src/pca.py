import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "./data.csv"

df = pd.read_csv(filename)

raw_data = df.values
cols = range(0, len(raw_data[0]))

"""
Data needs to be cleand up. In the dataset, values like "Sometimes" or "Car" are used.
Those need to be turned into numbers
"""
problematic_columns = [n for n, i in enumerate(raw_data[0]) if type(i) == str] # list of columns that are not numbers but strings

clean_data = raw_data

for i in problematic_columns:
    classLabels = raw_data[:, i]  # -1 takes the last feature
    classNames = np.unique(classLabels)
    classDict = dict(zip(classNames, range(len(classNames))))
    attributeNames = np.asarray(df.columns[cols])
    y = np.array([classDict[cl] for cl in classLabels]) # assigns numbers to the differen strings / values
    clean_data[:, i] = y

print(clean_data)

X = clean_data[:, cols]

N, M = X.shape

C = len(classNames)

# subtract the means from the featues
# Data is now centerd to the center of the coordinate system
Y = X - np.ones((N, 1)) * X.mean(axis=0) # axis = 0 is the vertical axis on a coordinate-system

plt.plot(Y[:, 0], Y[:, 2], "o")
plt.xlabel("Gender")
plt.ylabel("Height")
plt.title("Plot Gender agains Height")
plt.show()


print(Y)

