from sklearn import decomposition
from dataset import *


"""
This is just a test. I discoverd there is a buildin method in sklearn.
But I think our PCA is good too.
"""

pca = decomposition.PCA()
pca.fit(X)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(X)
print(data_transform)