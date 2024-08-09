# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Generate or load your dataset
data =pd.read_csv('/content/2022-06-09-enriched.csv')
# For example, let's create a dummy dataset for demonstration purposes
np.random.seed(42)
num_samples = 100
features = np.random.rand(num_samples, 2) * 10  # Two independent variables for demonstration
target = np.where(3 * features[:, 0] + 2 * features[:, 1] + np.random.randn(num_samples) * 2 > 15, 1, 0)  # Binary classification based on a threshold

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create an XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Plot the results (for binary classification)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='black', label='Actual Low Risk')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Actual High Risk')
plt.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], color='blue', marker='x', label='Predicted High Risk')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XGBoost Classifier for Cyber Risk Prediction')
plt.legend()
plt.show()