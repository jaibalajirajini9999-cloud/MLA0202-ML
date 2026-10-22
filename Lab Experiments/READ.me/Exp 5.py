import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Generate sample dataset
X, y = make_blobs(
    n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=1.3
)

# 2. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Create and train K-NN model
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# 4. Predict test data
y_pred = model.predict(X_test)

# 5. Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("K-NN Classification")
print("Number of Neighbours (K):", k)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# 6. Plot graph
plt.figure(figsize=(8, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="black",
    label="Training Data",
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="coolwarm",
    marker="^",
    s=100,
    edgecolors="black",
    label="Test Predictions",
)

plt.title("K-Nearest Neighbours Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
