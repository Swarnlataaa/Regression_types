# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate some example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Two independent variables
y = np.array([3, 5, 7, 8, 10])                          # Dependent variable

# Create a multiple linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict values for new data points
X_new = np.array([[6, 7], [7, 8]])
y_pred = model.predict(X_new)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted values for new data points:", y_pred)
