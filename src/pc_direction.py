from dataset import *


# SVD
U, S, Vh = svd(Y2, full_matrices=False)
V = Vh.T


"""
    the principal directions of the considered PCA components (either find a
    way to plot them or interpret them in terms of the features),
"""
# chosen pcs
pcs = range(0, 7)
legendStrs = ["PC" + str(e + 1) for e in pcs]
c = ["r", "g", "b"]
# bw = 0.2
bw = 0.11 # bar width
r = np.arange(1, M + 1)
for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)
plt.xticks(r + bw, attributeNames, rotation=45)
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Obesity: PCA Component Coefficients")
plt.show()