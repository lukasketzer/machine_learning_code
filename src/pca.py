import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

filename = "./data.csv"
df = pd.read_csv(filename)
rawData = df.values
cols = range(0, len(rawData[0]))


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


"""
Data needs to be cleand up. In the dataset, values like "Sometimes" or "Car" are used.
Those need to be turned into numbers
"""
# problematic_columns = [n for n, i in enumerate(rawData[0]) if type(i) == str] # list of columns that are not numbers but strings
cleanData = np.empty_like(rawData)
attributeNames = np.asarray(df.columns[cols])
labels = []

# TODO: code is still shit but kinda usefull (don't touch)
for i, n in enumerate(rawData[0]):
    classLabels = rawData[:, i]  # -1 takes the last feature

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

    labels.append(classDict)

X = cleanData[:, cols].astype(float)
N, M = X.shape

# Subtract the means from the featues
Y = X - np.ones((N, 1)) * X.mean(
    axis=0
)  # axis = 0 is the vertical axis on a coordinate-system

# Divide with standart derivation
Y2 = Y * (1 / np.std(Y, axis=0, ddof=1))

# SVD
U, S, Vh = svd(Y2, full_matrices=False)
V = Vh.T

"""
 Plot gender agains obesitity level
"""

plt.figure()
plt.plot(X[:, -1], X[:, 3], "o")
plt.title("Plotting Gender agains obestiy")
plt.xlabel("Gender")
plt.xticks(list(labels[-1].values()), list(labels[-1].keys()))
# plt.yticks(list(labels[3].values()), list(labels[3].keys())) # don't uncomment this
plt.ylabel("Obesity")
plt.grid()
plt.show()


"""
   The amount of variation explained as a function of the number of PCA
   components included, 
"""
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
plt.savefig("./figures/variance_principal_components.svg")
plt.show()


"""
    the principal directions of the considered PCA components (either find a
    way to plot them or interpret them in terms of the features),
"""
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
c = ["r", "g", "b"]
bw = 0.2
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("NanoNose: PCA Component Coefficients")
plt.show()


"""
    The data of a chosen arument projected onto the considered principal components.
"""
Z = Y @ V  # Project data on principal Compontes
pc1 = 0  # choseon principal componet
pc2 = 1
chosenAttribute = -1  # -1 is Obesity Level:w
classNamesAttribute = labels[chosenAttribute].keys()
# Plot PCA of the data
f = plt.figure()
plt.title("Obesitiy data: PCA")

for c in range(len(labels[chosenAttribute])):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, pc1], Z[class_mask, pc2], "o", alpha=0.5)
plt.legend(classNamesAttribute)
plt.xlabel("PC{0}".format(pc1 + 1))
plt.ylabel("PC{0}".format(pc2 + 1))

# Output result to screen
plt.savefig("./figures/data_onto_considered_principal_components.svg")
plt.show()
