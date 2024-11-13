from Dataset import Dataset
import numpy as np
from sklearn.model_selection import KFold



class ClassificationBaseline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Compute the largest class on the training data (most frequent class)
    def predict(self, X_test):
        classes = Dataset.sortedAttributes["NObeyesdad"]
        counts = [np.sum(self.y == n) for n in range(0, 7)]

        largest_class_index = counts.index(max(counts))
        largest_class = classes[largest_class_index]
        return np.full(X_test.shape[0], largest_class_index)



if __name__ == "__main__":
    seed = 20
    k = 10 # Number of folds for cross-validation

    cv = KFold(n_splits=k, shuffle=True, random_state=seed)

    dataset = Dataset()
    y = dataset.y # Target labels

    # Store the error rates for each fold
    error_rates = []

    # Perform k-fold cross-validation
    k = 1
    for train_index, test_index in cv.split(y):
        # Split the data into training and testing sets based on the current fold
        y_train, y_test = y[train_index], y[test_index]
        
        # Compute the largest class on the training data (most frequent class)
        mod = ClassificationBaseline(np.array([]), y_train)
        y_pred = mod.predict(y_test)

        # classes = dataset.sortedAttributes["NObeyesdad"]
        # counts = [np.sum(y_train == n) for n in range(0, 7)]

        # largest_class_index = counts.index(max(counts))
        # largest_class = classes[largest_class_index]
        
        # Calculate the number of miss-classifications by comparing predicted labels (largest class)
        # to the true labels in the test set
        miss_classifications = np.sum(y_pred != y_test)

        # Calculate the error rate for this fold
        error_rate = miss_classifications / len(y_test)
        print(f"{k} | {error_rate}")
        k += 1
        error_rates.append(error_rate)
    # Calculate and print the average error rate across all folds
