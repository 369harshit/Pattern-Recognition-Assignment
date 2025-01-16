import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the synthetic heart disease dataset
dataset_path = "synthetic_heart_disease_dataset.csv"
data = pd.read_csv(dataset_path)

# Modify the target to be continuous (for regression)
np.random.seed(42)
data["HeartDiseaseLikelihood"] = data["HeartDisease"] + np.random.normal(0, 0.1, len(data))

# Features and target
X = data[["Age", "Cholesterol", "MaxHeartRate"]]
y = data["HeartDiseaseLikelihood"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bayesian Ridge Regression
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
y_pred_bayesian = bayesian_model.predict(X_test)

# Ridge Regression (LS Regularization)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluation Metrics
mse_bayesian = mean_squared_error(y_test, y_pred_bayesian)
r2_bayesian = r2_score(y_test, y_pred_bayesian)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Display Metrics
print(f"Bayesian Ridge Regression - MSE: {mse_bayesian:.4f}, R2 Score: {r2_bayesian:.4f}")
print(f"Ridge Regression (LS Regularization) - MSE: {mse_ridge:.4f}, R2 Score: {r2_ridge:.4f}")

# Visualization
plt.figure(figsize=(10, 6))

# True Values vs Predictions
plt.plot(y_test.values, label="True Values", color="black", linestyle="dotted")
plt.plot(y_pred_bayesian, label="Bayesian Ridge Predictions", color="blue")
plt.plot(y_pred_ridge, label="Ridge Regression Predictions", color="red")

plt.legend()
plt.title("Regression: Bayesian Ridge vs Ridge Regression")
plt.xlabel("Sample Index")
plt.ylabel("Heart Disease Likelihood (Continuous)")
plt.show()

"""Dataset:

The program uses the synthetic heart disease dataset with features:
Age, Cholesterol, and MaxHeartRate.
Target: HeartDisease (binary classification: 0 = no heart disease, 1 = heart disease).
Preprocessing:

The dataset is split into training and testing sets.
Features are standardized using StandardScaler for better model performance.
Models:

Bayesian Ridge Regression: Incorporates priors to prevent overfitting.
Ridge Regression (LS Regularization): Penalizes large coefficients to manage overfitting.
Evaluation Metrics:

Mean Squared Error (MSE): Measures prediction error.
R² Score: Represents the proportion of variance explained by the model.
Visualization:

Compares true HeartDisease values with predictions from both models.
"""