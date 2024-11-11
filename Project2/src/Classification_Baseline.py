from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold

seed = 20
k = 10 # Number of folds for cross-validation

cv = KFold(n_splits=k, shuffle=True, random_state=seed)

dataset = Dataset()
y = dataset.y # Target labels

# Store the error rates for each fold
error_rates = []

# Perform k-fold cross-validation
for train_index, test_index in cv.split(y):
    # Split the data into training and testing sets based on the current fold
    y_train, y_test = y[train_index], y[test_index]
    
    # Compute the largest class on the training data (most frequent class)
    classes = dataset.sortedAttributes["NObeyesdad"]
    counts = [np.sum(y_train == n) for n in range(0, 7)]
    largest_class_index = counts.index(max(counts))
    largest_class = classes[largest_class_index]
    
    # Calculate the number of miss-classifications by comparing predicted labels (largest class)
    # to the true labels in the test set
    miss_classifications = np.sum(largest_class_index != y_test)
    
    # Calculate the error rate for this fold
    error_rate = miss_classifications / len(y_test)
    error_rates.append(error_rate)
    
for i in range(1, k + 1):
    print(f"{i}: {100*error_rates[i-1]:.4f}%")

# Calculate and print the average error rate across all folds
average_error_rate = np.mean(error_rates)
print(f"Average error rate across {k} folds: {100*average_error_rate:.4f}%")
