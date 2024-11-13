from sklearn.linear_model import LogisticRegression
from Dataset import Dataset
import numpy as np
import warnings
import pandas as pd

# Set seed for reproducibility
seed = 20
lambda_value = 1e-5  # Predefined lambda value (regularization parameter)

# Load the dataset
dataset = Dataset()
X = np.delete(dataset.X_mean_std, 11, 1)  # Remove NObeyesidad attribute
y = dataset.y  # Corresponding target output

# Suppress warnings temporarily
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    # Initialize and fit the multinomial logistic regression model on the entire dataset
    logreg = LogisticRegression(
        C=1.0 / lambda_value,  # Set C as the inverse of predefined lambda
        solver="lbfgs", multi_class="multinomial", tol=1e-3, random_state=seed
    )
    logreg.fit(X, y)

# Get feature names (excluding the last attribute if necessary)
feature_names = dataset.attributeNames[:-1]  # Assuming the last column is not relevant

# Get the weights (coefficients) for each class
final_weights = logreg.coef_

# Create a DataFrame to store the weights with features as rows and classes as columns
weights_df = pd.DataFrame(final_weights.T, columns=[f"Class {i}" for i in range(final_weights.shape[0])], index=feature_names)

# Print the table of feature weights for each class
print("\nFeature weights for each class:")
print(weights_df)
