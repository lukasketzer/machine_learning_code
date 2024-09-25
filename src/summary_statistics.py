from dataset import *
import numpy as np


for count, item in enumerate(attributeNames):
    data = X[:, count]

    # Compute values
    mean_x = data.mean()
    std_x = data.std(ddof=1)
    median_x = np.median(data)
    range_x = data.max() - data.min()

    # Display results
    print("================================")
    print("Attribute:", item)
    print("Mean:", mean_x)
    print("Standard Deviation:", std_x)
    print("Median:", median_x)
    print("Range:", range_x)

