from dataset import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



figs, ax = plt.subplots(4, 3)
# figs, ax = plt.subplots(3, 4)

for i, n in enumerate(attributeNames):
    Y2 = Y2[:ORIGINAL_DATA_COUNT]
    data = X[:, i]
    N = len(data)

    nbins = min(N, 20)

    mean = np.mean(data)
    s =  np.std(data, ddof=1)
    print(f"mean: {mean}, s {s}")

    ax.flat[i].set_title(n)
    # ax.flat[i].hist(data, bins=nbins, edgecolor="white")
    ax.flat[i].hist(data, density=True, bins='auto', histtype='stepfilled', alpha=0.6)

    if labels[i] != {}:
        ax.flat[i].set_xticks(list(labels[i].values()), list(labels[i].keys()), rotation=45)
        
    match n:
        case "Age":
            ax.flat[i].set_xlabel("years")
        case "Height":
            ax.flat[i].set_xlabel("meters")
        case "Weight":
            ax.flat[i].set_xlabel("kg")

    # theoretical normal distribution
    x = np.linspace(data.min(), data.max(), 5000)
    pdf = stats.norm.pdf(x, mean, s)
    ax.flat[i].plot(x, pdf, color="red")

plt.subplots_adjust(wspace=0.5, hspace=0.7)
plt.savefig("./figures/normal_distribution.svg")
plt.savefig("./figures/normal_distribution.eps")
plt.show()
