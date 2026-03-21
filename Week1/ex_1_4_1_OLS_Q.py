#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:06:27 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Learning Objective: Observe OLS behavior in the Stable Regime (n >> m).

# 1. Load Data
print('Loading Wine Quality data...')
# TASK: Load wine-quality-red, scale features, and cast target to float.
data = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
X = data.data
y = data.target.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)
print('Data loaded and scaled.')
# 2. Stable Split
# TASK: Create a split with 80% training data (Large n, Small m).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit OLS
# TASK: Use LinearRegression.
model = LinearRegression().fit(X_train, y_train)


# 4. Evaluate
# TASK: Calculate and print Training MSE and Test MSE.
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 5. Visualization
# TASK: Create a bar chart comparing Train vs Test MSE.

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Training MSE: {train_mse:.4f}')
print(f'Test MSE: {test_mse:.4f}')
#Bar chart
labels = ['Train MSE', 'Test MSE']
mse_values = [train_mse, test_mse]
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.ylabel('Mean Squared Error')
plt.title('Train vs Test MSE for OLS on Wine Quality Data')
plt.show()
