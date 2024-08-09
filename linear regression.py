# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

# Generate or load your dataset
data = pd.read_csv('/content/2022-06-09-enriched.csv')
# For example, let's create a dummy dataset for demonstration purposes
np.random.seed(42)
num_samples = 100
features = np.random.rand(num_samples, 1) * 10  # Independent variable
target = 3 * features + 5 + np.random.randn(num_samples, 1) * 2  # Dependent variable with some noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Convert regression to classification
threshold = 7  # Adjust the threshold based on your problem
y_pred_class = np.where(y_pred > threshold, 1, 0)
y_test_class = np.where(y_test > threshold, 1, 0)

# Calculate accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f'Accuracy: {accuracy}')

# Plot the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Linear Regression Model')
plt.show()