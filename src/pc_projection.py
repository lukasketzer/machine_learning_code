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

Z = Y @ V  # Project data on principal Compontes

"""
    The data of a chosen arument projected onto the considered principal components.
"""
# Plot PCA of the data
pc1 = 0  # chosen principal componet
pc2 = 1
pc3 = 2

f = plt.figure()
plt.title("Obesitiy data: PCA")
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, pc1], Z[class_mask, pc2], "o")
plt.legend(classNames)
plt.xlabel("PC{0}".format(pc1 + 1))
plt.ylabel("PC{0}".format(pc2 + 1))

# Output result to screen
plt.savefig("./figures/data_onto_considered_principal_components.svg")
plt.savefig("./figures/data_onto_considered_principal_components.eps")
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.title("Obesity Data: PCA")

# # Plot data points for each class
# for c in range(C):
#     # Select indices belonging to class c
#     class_mask = y == c
#     ax.scatter(Z[class_mask, pc1], Z[class_mask, pc2], Z[class_mask, pc3], label=classNamesAttribute)

# # Add legend, labels, and title
# ax.legend(classNames, loc='best')
# ax.set_xlabel("PC{0}".format(pc1 + 1))
# ax.set_ylabel("PC{0}".format(pc2 + 1))
# ax.set_zlabel("PC{0}".format(pc3 + 1))

# # Output result to screen and save the plot
# plt.show()

