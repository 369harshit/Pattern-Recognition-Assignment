from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Diabetes dataset (a regression dataset)
diabetes = load_diabetes()
X, y = diabetes.data[:, 2:3], diabetes.target  # Using a single feature for easy visualization

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linear SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_scaled, y)

# Non-Linear SVR with RBF kernel
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_scaled, y)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot for Linear SVR
plt.subplot(1, 2, 1)
plt.scatter(X_scaled, y, color='gray', label='Data')
plt.plot(X_scaled, linear_svr.predict(X_scaled), color='red', label='Linear SVR')
plt.title("Linear SVM Regression")
plt.legend()

# Plot for RBF SVR
plt.subplot(1, 2, 2)
plt.scatter(X_scaled, y, color='gray', label='Data')
plt.plot(X_scaled, rbf_svr.predict(X_scaled), color='blue', label='RBF SVR')
plt.title("RBF SVM Regression")
plt.legend()

plt.show()
