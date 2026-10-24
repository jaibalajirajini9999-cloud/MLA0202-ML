import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_poly = model_poly.predict(X_poly)

# Display results
print("Linear Regression R2 Score:", round(r2_score(y, y_linear), 2))
print("Polynomial Regression R2 Score:", round(r2_score(y, y_poly), 2))

# Plot graphs
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_linear, color="red", label="Linear Regression")
plt.plot(X, y_poly, color="green", label="Polynomial Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()
plt.show()
