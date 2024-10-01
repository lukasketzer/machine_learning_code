from dataset import *
import seaborn as sns
import pandas as pd

# SVD
# Y2 = Y2[:ORIGINAL_DATA_COUNT]
U, S, Vh = svd(Y2, full_matrices=False)
V = Vh.T

Z = Y @ V  # Project data on principal Compontes

"""
    The data of a chosen arument projected onto the considered principal components.
"""

pcs = 5
d = pd.DataFrame(Z[:, :pcs], columns=[f"PC {i + 1}" for i in range(pcs)])
d.insert(len(d.columns), "NObeyesdad", classLabels, True)
sns.pairplot(
    d,
    corner=True,
    hue="NObeyesdad",
    diag_kind="hist",
    hue_order=sortedAttributes["NObeyesdad"],
    # palette=sns.color_palette("tab10", pcs),
)
plt.show()
