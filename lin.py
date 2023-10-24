# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
y = np.array([2, 4, 5, 4, 5])              # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict values for new data points
X_new = np.array([6, 7]).reshape(-1, 1)
y_pred = model.predict(X_new)

# Plot the data and regression line
plt.scatter(X, y, label='Actual Data')
plt.plot(X, model.predict(X), label='Regression Line', color='red')
plt.scatter(X_new, y_pred, label='Predicted Data', color='green')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.show()
