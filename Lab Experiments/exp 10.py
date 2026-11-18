import numpy as np
from sklearn.mixture import GaussianMixture

# Sample data
X = np.array([[1], [2], [3], [8], [9], [10]])

# Initialize EM model
model = GaussianMixture(n_components=2, random_state=0)

# Train the model
model.fit(X)

# Predict clusters
labels = model.predict(X)

# Display results
print("Cluster Labels:", labels)
print("Means:", model.means_)
print("Covariances:", model.covariances_)
print("Weights:", model.weights_)
