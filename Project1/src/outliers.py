import numpy as np
from matplotlib.pyplot import (
    boxplot,
    figure,
    hist,
    show,
    subplot,
    subplots,
    title,
    xlabel,
    xticks,
    ylim,
    yticks,
)

from dataset import *
font = {'size'   : 22}

matplotlib.rc('font', **font)

fig, axs = subplots(nrows=1, ncols=2)

continuous = [1, 2, 3]

axs.flat[0].set_title("Obesity: Boxplot (standardized)")
axs.flat[0].boxplot([Y2[:, c] for c in continuous])
axs.flat[0].set_xticks(continuous, [attributeNames[c] for c in continuous], rotation=45)

axs.flat[1].set_title("Obesity: Boxplot (regular)")
axs.flat[1].boxplot([X[:, c] for c in continuous])
axs.flat[1].set_xticks(continuous, ["Age\nin years", "Height\nin meters", "Weight\nin kg"], rotation=45)
show()
exit(1)

