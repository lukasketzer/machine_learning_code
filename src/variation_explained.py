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
   The amount of variation explained as a function of the number of PCA
   components included, 
"""
# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()
# print(f"S: {S}")
# print(f"Rho: {rho}")

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), [0.9 for _ in rho], "-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative"])
plt.grid()
plt.savefig("./figures/variance_principal_components.svg")
plt.savefig("./figures/variance_principal_components.eps")
plt.show()

