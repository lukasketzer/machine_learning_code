import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

filename = "./data.csv"
df = pd.read_csv(filename)
df = df.drop("SMOKE")
df = df.drop("FAVC")
df = df.drop("FCVC")
# df = df.drop("NCP")
df = df.drop("CAEC")

rawData = df.values
cols = range(0, len(rawData[0]))
ORIGINAL_DATA_COUNT = 485


# Sorted the occurences of the attributes in a logical order
sortedAttributes = {
    "family_history_with_overweight": ["yes", "no"],
    "FAVC": ["yes", "no"],
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "SMOKE": ["yes", "no"],
    "SCC": ["yes", "no"],
    "TUE": ["0—2 hours", "3—5 hours", "More than 5 hours"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
    "MTRANS": ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
    "NObeyesdad": [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ],
}
attributeNames = np.asarray(df.columns[cols])

classLabels = rawData[:, -1]
classNames = sortedAttributes["NObeyesdad"] 
classDict = dict(zip(classNames, range(len(classNames))))
C = len(classNames)

y = np.array([classDict[cl] for cl in classLabels])


X_raw = rawData[:, cols]


"""
Data needs to be cleand up. In the dataset, values like "Sometimes" or "Car" are used.
Those need to be turned into numbers
"""
# problematic_columns = [n for n, i in enumerate(rawData[0]) if type(i) == str] # list of columns that are not numbers but strings
cleanData = np.empty_like(rawData)

for i, n in enumerate(rawData[0]):
    classLabels = rawData[:, i]  # 

    if attributeNames[i] in sortedAttributes.keys():
        classNames = sortedAttributes[attributeNames[i]]
    else:
        classNames = np.unique(classLabels)

    if type(n) != str:
        cleanData[:, i] = rawData[:, i]
        # classDict = dict(zip(classNames, classNames))
        classDict = {}
    else:
        classDict = dict(zip(classNames, range(len(classNames))))
        y = np.array(
            [classDict[cl] for cl in classLabels]
        )  # assigns numbers to the differen strings / values
        cleanData[:, i] = y


X = cleanData[:, cols].astype(float)
N, M = X.shape

# TODO: maybe move in own file
# Subtract the means from the featues
Y = X - np.ones((N, 1)) * X.mean(
    axis=0
)  # axis = 0 is the vertical axis on a coordinate-system

# normalization
Y2 = Y * (1 / np.std(Y, axis=0, ddof=1))