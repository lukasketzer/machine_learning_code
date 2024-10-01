import numpy as np
from matplotlib.pyplot import (
    boxplot,
    figure,
    hist,
    show,
    subplot,
    title,
    xlabel,
    xticks,
    ylim,
    yticks,
)

from dataset import *
font = {'size'   : 22}

matplotlib.rc('font', **font)

figure()
title("Obesity: Boxplot")
continuous = [1, 2, 3]
boxplot([Y2[:, c] for c in continuous])
xticks(continuous, [attributeNames[c] for c in continuous], rotation=45)
show()
exit(1)

