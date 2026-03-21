#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:49:11 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- SECTION 1: Data Loading & Expansion ---
print('Fetching Wine Quality data...')
# TASK: Load the 'wine-quality-red' dataset.
# TASK: Use PolynomialFeatures(degree=2) to expand the feature set.
# This increases m (parameters) significantly.

# df = ...
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = ...

# --- SECTION 2: Preprocessing for Overfitting ---
# TASK: 
# 1. Scale the expanded features.
# 2. Perform a split where the training set is very small (e.g., test_size=0.8).
# This creates the n << m regime.

# X_scaled = ...
# X_train, X_test, y_train, y_test = ...

# print(f'Training samples (n): {X_train.shape[0]}')
# print(f'Features (m): {X_train.shape[1]}')

# --- SECTION 3: Complexity Sweep ---
# TASK: Sweep through Ridge alpha values (lambdas).
# Use a wide range, e.g., np.logspace(-3, 5, 50).

lambdas = np.logspace(-3, 5, 50)
train_errors = []
test_errors = []

# for l in lambdas:
#    model = Ridge(alpha=l).fit(X_train, y_train)
#    ...

# --- SECTION 4: Visualization ---
# TASK: Plot Training vs. Test Error.
# 1. Invert the x-axis (complexity increases to the right).
# 2. Use plt.ylim() to focus on the 'valley' of the U-shape if the error explodes.

# plt.figure(figsize=(10, 6))
# ...
# plt.gca().invert_xaxis()
# plt.show()