from dataset import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Number of samples
N = 200


figs, ax = plt.subplots(4, 5)

for i, n in enumerate(attributeNames):
    # Y2 = Y2[:ORIGINAL_DATA_COUNT]
    data = Y2[:, i]
    N = len(data)

    nbins = min(N, 20)

    mean = np.mean(data)
    s =  np.std(data, ddof=1)
    print(f"mean: {mean}, s {s}")

    ax.flat[i].set_title(n)
    # ax.flat[i].hist(data, bins=nbins, edgecolor="white")
    ax.flat[i].hist(data, density=True, bins='auto', histtype='stepfilled', alpha=0.6)

    # theoretical normal distribution
    x = np.linspace(data.min(), data.max(), 5000)
    pdf = stats.norm.pdf(x, mean, s)
    ax.flat[i].plot(x, pdf, color="red")

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
