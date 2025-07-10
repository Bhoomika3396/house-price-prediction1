# 🏡 House Price Prediction - Boston Housing Dataset

This project is part of my Machine Learning Internship at Bharat Intern.

## 📊 Objective
Predict house prices using the Boston Housing Dataset by training regression models.

## 📁 Dataset
- Source: `sklearn.datasets.load_boston()` (Scikit-learn v0.23)
- Features include crime rate, number of rooms, tax rate, etc.

## 🧪 ML Pipeline

- Data Loading using Scikit-learn
- Preprocessing:
  - Feature scaling (StandardScaler)
  - Train-Test split (80-20)
- Models:
  - Linear Regression
  - Decision Tree Regressor
- Evaluation:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Visualization:
  - Actual vs Predicted Scatter Plots

## 🛠️ How to Run

1. Clone the repository
2. Install required packages:

```bash
pip install -r requirements.txt
