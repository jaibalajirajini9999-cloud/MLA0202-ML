import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)

X1 = np.random.normal(2, 0.5, (100, 2))
X2 = np.random.normal(5, 0.5, (100, 2))
X = np.vstack((X1, X2))

# Apply EM algorithm using Gaussian Mixture Model
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Expectation-Maximization Algorithm")
print("Number of clusters:", model.n_components)
print("Converged:", model.converged_)

# Plot graph
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels, cmap="viridis"
)

plt.scatter(
    model.means_[:, 0], model.means_[:, 1],
    color="red", marker="X", s=200,
    label="Cluster Centers"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("EM Algorithm - Gaussian Mixture Model")
plt.legend()
plt.grid()
plt.show()
