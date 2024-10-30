import numpy as np
import pandas as pd
from typing import List
import pathlib
import matplotlib.pyplot as plt
from scipy.linalg import svd

class Dataset:
    filename = f"{pathlib.Path(__file__).parent.resolve()}/../../data/data.csv"
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
        "MTRANS": [
            "Automobile",
            "Bike",
            "Motorbike",
            "Public_Transportation",
            "Walking",
        ],
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


    def __init__(self, original_data: bool = False):
        df = pd.read_csv(self.filename)
        df = df.drop("SMOKE", axis=1)
        df = df.drop("FAVC", axis=1)
        df = df.drop("FCVC", axis=1)
        # df = df.drop("NCP")
        df = df.drop("CAEC", axis=1)
        df = df.drop("SCC", axis=1)
        # df = df.drop("TUE", axis=1)

        if (original_data):
            rawData = df.values[:self.ORIGINAL_DATA_COUNT]
        else:
            rawData = df.values
        cols = range(0, len(rawData[0]))

        attributeNames = np.asarray(df.columns[cols]) # names of the columns

        classLabels = rawData[:, -1] # does not need to be public
        classNames = self.sortedAttributes["NObeyesdad"]
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
        labels = []

        for i, n in enumerate(rawData[0]):
            classLabels = rawData[:, i]  #

            if attributeNames[i] in self.sortedAttributes.keys():
                classNames = self.sortedAttributes[attributeNames[i]]
            else:
                classNames = np.unique(classLabels)

            if type(n) != str:
                cleanData[:, i] = rawData[:, i]
                # classDict = dict(zip(classNames, classNames))
                classDict = {}

            else:
                classDict = dict(zip(classNames, range(len(classNames))))
                y2 = np.array(
                    [classDict[cl] for cl in classLabels]
                )  # assigns numbers to the differen strings / values
                cleanData[:, i] = y2

            labels.append(classDict)

        X = cleanData[:, cols].astype(float)
        N, M = X.shape

        # Subtract the means from the featues
        X_mean = X - np.ones((N, 1)) * X.mean(
            axis=0
        )  # axis = 0 is the vertical axis on a coordinate-system

        # standardization
        X_mean_std = X_mean * (1 / np.std(X_mean, axis=0, ddof=1))

        self.X = X
        self.X_mean = X_mean
        self.X_mean_std = X_mean_std
        self.X_raw = X_raw 
        self.y = y
        self.N = N
        self.M = M
        self.C = C
        self.attributeNames = attributeNames
        self.classNames = classNames
        self.classDict = classDict


if __name__ == "__main__":
    dataset = Dataset(original_data=True)
    # SVD
    U, S, Vh = svd(dataset.X_mean_std, full_matrices=False)
    V = Vh.T

    """
    The amount of variation explained as a function of the number of PCA
    components included, 
    """
    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()
    # print(f"S: {S}")
    # print(f"Rho: {rho}")

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative")
    plt.plot(range(1, len(rho) + 1), [0.7 for _ in rho], label="Threshold", color="gray", linestyle="dashed", lw=2.0)
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    # plt.legend(["Individual", "Cumulative"])
    plt.legend()
    plt.grid()
    plt.show()


