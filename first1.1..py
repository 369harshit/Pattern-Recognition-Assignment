import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data[:, :2]  # Use the first two features for simplicity
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the dataset to a CSV file
dataset = pd.DataFrame(X, columns=['SepalLength', 'SepalWidth'])
dataset['Label'] = y
dataset.to_csv('iris_dataset.csv', index=False)

# Underfitting model: Logistic Regression
underfit_model = LogisticRegression()
underfit_model.fit(X_train, y_train)

# Overfitting model: Decision Tree with high depth
overfit_model = DecisionTreeClassifier(max_depth=10)
overfit_model.fit(X_train, y_train)

# Balanced model: Decision Tree with moderate depth
balanced_model = DecisionTreeClassifier(max_depth=3)
balanced_model.fit(X_train, y_train)

# Accuracy Scores
underfit_train_acc = accuracy_score(y_train, underfit_model.predict(X_train))
underfit_test_acc = accuracy_score(y_test, underfit_model.predict(X_test))

overfit_train_acc = accuracy_score(y_train, overfit_model.predict(X_train))
overfit_test_acc = accuracy_score(y_test, overfit_model.predict(X_test))

balanced_train_acc = accuracy_score(y_train, balanced_model.predict(X_train))
balanced_test_acc = accuracy_score(y_test, balanced_model.predict(X_test))

# Visualization function
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)

# Plot decision boundaries
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plot_decision_boundary(underfit_model, X, y, "Underfitting: Logistic Regression")

plt.subplot(1, 3, 2)
plot_decision_boundary(overfit_model, X, y, "Overfitting: Decision Tree (High Depth)")

plt.subplot(1, 3, 3)
plot_decision_boundary(balanced_model, X, y, "Balanced Fit: Decision Tree (Moderate Depth)")

plt.tight_layout()
plt.show()

# Bias-Variance tradeoff analysis
print("Dataset saved as 'iris_dataset.csv'")
print("Underfitting Model: Train Accuracy =", underfit_train_acc, ", Test Accuracy =", underfit_test_acc)
print("Overfitting Model: Train Accuracy =", overfit_train_acc, ", Test Accuracy =", overfit_test_acc)
print("Balanced Model: Train Accuracy =", balanced_train_acc, ", Test Accuracy =", balanced_test_acc)

# Learning Curve for Bias-Variance Trade-off
def plot_learning_curve(model, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label="Training Accuracy")
    plt.plot(train_sizes, test_mean, label="Testing Accuracy")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

# Plot learning curves for all three models
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plot_learning_curve(underfit_model, "Learning Curve: Logistic Regression (Underfitting)")

plt.subplot(2, 2, 2)
plot_learning_curve(overfit_model, "Learning Curve: Decision Tree (Overfitting)")

plt.subplot(2, 2, 3)
plot_learning_curve(balanced_model, "Learning Curve: Decision Tree (Balanced)")

plt.tight_layout()
plt.show()



"""Label: 0, 1, 2
These numbers (0, 1, and 2) represent the type (species) of the flower:

0: Setosa – A type of iris flower with distinct features.
1: Versicolor – Another type of iris flower with slightly different features.
2: Virginica – A third type of iris flower with its own characteristics.
The goal of the classification is to use the measurements (Sepal Length and Sepal Width) to predict which flower (Setosa, Versicolor, or Virginica) it is."""