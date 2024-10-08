from dataset import *


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
plt.plot(range(1, len(rho) + 1), rho, "x-", label="Individual")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", label="Cumulative")
plt.plot(range(1, len(rho) + 1), [0.7 for _ in rho], label="Threshold", color="gray", linestyle="dashed", lw=2.0)
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
# plt.legend(["Individual", "Cumulative"])
plt.legend()
plt.grid()
plt.savefig("./figures/variance_principal_components.svg")
plt.savefig("./figures/variance_principal_components.eps")
plt.show()

