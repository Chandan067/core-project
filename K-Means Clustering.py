# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate or load your dataset
# For example, let's create a dummy dataset for demonstration purposes
np.random.seed(42)
num_samples = 100
features = np.random.rand(num_samples, 2) * 10  # Two independent variables for demonstration
true_labels = np.where(3 * features[:, 0] + 2 * features[:, 1] + np.random.randn(num_samples) * 2 > 15, 1, 0)  # Binary classification based on a threshold

# Create a K-means clustering model
num_clusters = 2
model = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model to the data and predict clusters
cluster_labels = model.fit_predict(features)

# Map cluster labels to binary values (0 or 1) based on majority class
cluster_mapping = {cluster: np.argmax(np.bincount(true_labels[cluster_labels == cluster])) for cluster in range(num_clusters)}
predicted_labels = np.array([cluster_mapping[label] for label in cluster_labels])

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy}')

# Visualize the clusters
plt.scatter(features[:, 0], features[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, marker='X', c='red', label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering with Accuracy for Cyber Risk Prediction')
plt.legend()
plt.show()