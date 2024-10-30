from Dataset import Dataset
import numpy as np

dataset = Dataset()
y = dataset.y # target output

np.random.seed(42)
num_samples = y.shape[0]
shuffled_indices = np.random.permutation(num_samples)

y = y[shuffled_indices]
y_train = y[:int(0.8 * dataset.N)] # y = labels (target output)
y_test = y[int(0.8 * dataset.N):]

# Compute the largest class on the training data
classes = dataset.sortedAttributes["NObeyesdad"]
counts = [np.sum(y_train == n) for n in range(0, 7)]
largest_class_index = counts.index(max(counts))
largest_class = classes[largest_class_index]

# Number of miss-classifications
print(
    "Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}".format(
        np.sum(largest_class_index != y_test), len(y_test)
    )
)
