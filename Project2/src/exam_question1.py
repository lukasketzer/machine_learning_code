# %%
import numpy as np
import matplotlib.pyplot as plt

#fpr = [1,.5, 2/3, 2/4,3/5,3/6,4/7,4/8]
#tpr = [0,.5, 1/3, 2/4, 2/5, 3/6, 3/7, 4/8]
fpr = np.array([0,1,2,2,2,3,4,4,4]) / 4
tpr = np.array([0,0,0,1,2,2,2,3,4]) / 4
plt.title("B")
plt.plot(fpr, tpr)
plt.show()

# %%
fpr = np.array([0,1,1,2,2,2,2,3,4]) / 4
tpr = np.array([0,0,1,1,2,3,4,4,4]) / 4
plt.title("A")
plt.plot(fpr, tpr)
plt.show()

# %%
# ROC curve for C
tpr = np.array([0,1,1,1,2,3,3,3,4]) / 4
fpr = np.array([0,0,1,2,2,2,3,4,4]) / 4
plt.ylabel("TPR")
plt.xlabel("FPR")
print(tpr)
print(fpr)

#plt.title("C")
plt.plot(fpr, tpr)
plt.show()

# %%
fpr = np.array([0,1,1,2,2,3,3,4,4]) / 4
tpr = np.array([0,0,1,1,2,2,3,3,4]) / 4
print(fpr)
print(tpr)
plt.title("D")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.plot(fpr, tpr)
plt.show()


