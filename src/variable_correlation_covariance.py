from dataset import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

print(attributeNames)
np.set_printoptions(threshold=sys.maxsize, precision=2)
covarianceMatrix = np.cov(Y2, rowvar=0, ddof=1)
# print(covarianceMatrix)
correlationMatrix = np.corrcoef(Y2, rowvar=0, ddof=1)
print(correlationMatrix)
# # print(covarianceMatrix)
# d = pd.DataFrame(Y2[:ORIGINAL_DATA_COUNT, :5])
# sns.pairplot(d)
# plt.show()


