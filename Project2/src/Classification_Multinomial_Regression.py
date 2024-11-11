from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from Dataset import Dataset
import numpy as np
import sklearn.linear_model as lm
import warnings

seed = 20

# Lambda values to test
lambda_values = [0.01, 0.1, 1, 10, 100, 200, 500, 1000, 2000, 5000]

# Parameters for nested cross-validation
outer_k = 10   # Number of outer folds
inner_k = 10   # Number of inner folds

# Outer cross-validation loop
outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=seed)
outer_errors = []  # To store the final error rate for each outer fold

dataset = Dataset()
X = dataset.X_mean_std # Input data
y = dataset.y # Corresponding target output

for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    # print(f"\nStarting Outer Fold {outer_fold_idx}/{outer_k}")
    print(f"\033[1;30;44m\nStarting Outer Fold {outer_fold_idx}/{outer_k}\033[0m")

    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner cross-validation loop for hyperparameter tuning
    inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
    inner_errors = []  # To store the error rate for each lambda in the inner loop
    
    for lambda_value in lambda_values:
        # print(f"\n  Evaluating Lambda Value: {lambda_value}")
        fold_errors = []  # To store the error for each inner fold with this lambda
        
        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer), start=1):
            # print(f"    Inner Fold {inner_fold_idx}/{inner_k} for Lambda = {lambda_value}")
            X_train_inner, X_val_inner = X_train_outer[inner_train_idx], X_train_outer[inner_val_idx]
            y_train_inner, y_val_inner = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]
            
            # Suppress FutureWarnings temporarily
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                # Initialize and fit the multinomial logistic regression model
                logreg = LogisticRegression(
                    C=1.0 / lambda_value,  # Set C as the inverse of lambda
                    solver="lbfgs", multi_class="multinomial", tol=1e-3, random_state=seed
                )
                logreg.fit(X_train_inner, y_train_inner)
            
            # Calculate error rate on the validation set
            y_val_pred = logreg.predict(X_val_inner)
            error_rate = 1 - accuracy_score(y_val_inner, y_val_pred)
            fold_errors.append(error_rate)
            # print(f"      Validation Error Rate for Inner Fold {inner_fold_idx}: {100*error_rate:.4f}%")
        
        # Average error rate across inner folds for this lambda
        avg_inner_error = np.mean(fold_errors)
        inner_errors.append((lambda_value, avg_inner_error))
        # print(f"  Average Inner CV Error Rate for Lambda {lambda_value}: {100*avg_inner_error:.4f}%")

    # Select the lambda with the lowest average error in the inner loop
    best_lambda, best_inner_error = min(inner_errors, key=lambda x: x[1])
    print(f"\nBest Lambda for Outer Fold {outer_fold_idx}: λ={best_lambda}")
    
    # Suppress FutureWarnings temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Retrain the model with the best lambda on the entire outer training set
        best_logreg = LogisticRegression(
            C=1.0 / best_lambda, solver="lbfgs", multi_class="multinomial", tol=1e-3, random_state=seed
        )
        best_logreg.fit(X_train_outer, y_train_outer)
    
    # Evaluate on the outer test set
    y_test_outer_pred = best_logreg.predict(X_test_outer)
    outer_error = 1 - accuracy_score(y_test_outer, y_test_outer_pred)
    outer_errors.append(outer_error)

    # Calculate the number of misclassifications and total number of values in the test set
    num_misclassifications = np.sum(y_test_outer != y_test_outer_pred)
    total_test_values = len(y_test_outer)

    # Print the error rate, number of misclassifications, and total values
    print(f"Outer Fold {outer_fold_idx} Error Rate with Best Lambda (λ={best_lambda}): {100*outer_error:.4f}%")
    print(f"Number of misclassifications: {num_misclassifications} out of {total_test_values} samples")

# Final evaluation: average error across outer folds
final_outer_error = np.mean(outer_errors)
print(f"\nAverage Error Rate across all Outer Folds: {100*final_outer_error:.4f}%")
