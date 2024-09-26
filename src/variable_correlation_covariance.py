from dataset import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

covarianceMatrix = np.cov(Y2)
print(covarianceMatrix)
d = pd.DataFrame(Y2[:ORIGINAL_DATA_COUNT, :5])
sns.pairplot(d)
plt.show()


