from dataset import *


# Subtract the means from the featues
Y = X - np.ones((N, 1)) * X.mean(
    axis=0
)  # axis = 0 is the vertical axis on a coordinate-system


# normalization
Y2 = Y * (1 / np.std(Y, axis=0, ddof=1))

# SVD
U, S, Vh = svd(Y2, full_matrices=False)
V = Vh.T


"""
    the principal directions of the considered PCA components (either find a
    way to plot them or interpret them in terms of the features),
"""
# chosen pcs
pcs = range(0, 2)
legendStrs = ["PC" + str(e + 1) for e in pcs]
c = ["r", "g", "b"]
bw = 0.2
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Obesity: PCA Component Coefficients")
plt.show()