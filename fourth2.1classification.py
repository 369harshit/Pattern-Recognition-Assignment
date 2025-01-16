import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = datasets.load_iris()
X = data.data[:, :2]  # Using only the first two features for simplicity
y = data.target

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Non-Linear SVM with RBF kernel
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Evaluate accuracy
print("Linear SVM Accuracy: ", accuracy_score(y_test, y_pred_linear))
print("RBF SVM Accuracy: ", accuracy_score(y_test, y_pred_rbf))

# Plot decision boundaries
def plot_decision_boundary(X, y, model, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# Plot decision boundaries for Linear and RBF SVM
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(X, y, linear_svm, "Linear SVM Decision Boundary")

plt.subplot(1, 2, 2)
plot_decision_boundary(X, y, rbf_svm, "RBF SVM Decision Boundary")
