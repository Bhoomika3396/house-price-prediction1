from sklearn.datasets import load_boston
import pandas as pd

# Load dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name="MEDV")

# Inspect the data
print(X.head())
print(f"Shape: {X.shape}")
print(f"Target variable: {y.name}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train_scaled, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)

# Evaluation
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)

print(f"\nLinear Regression - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
print(f"Decision Tree - MAE: {mae_dt:.2f}, RMSE: {rmse_dt:.2f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Linear Regression")
plt.scatter(y_test, y_pred_dt, alpha=0.5, label="Decision Tree", color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
