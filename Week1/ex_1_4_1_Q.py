#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:47:23 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- SECTION 1: Data Loading ---
print('Fetching Wine Quality data...')
# TASK: Load the 'wine-quality-red' dataset from OpenML.
# Ensure the target is cast to float.

# df = ...
# X = ...
# y = ...

# --- SECTION 2: Preprocessing ---
# TASK: 
# 1. Scale the features using StandardScaler.
# 2. Perform a standard 70/30 train/test split.

# X_scaled = ...
# X_train, X_test, y_train, y_test = ...

# --- SECTION 3: Complexity Sweep ---
# TASK: Sweep through different values of lambda (Ridge alpha).
# Calculate both Training MSE and Test MSE for each.

lambdas = np.logspace(-3, 5, 25)
train_errors = []
test_errors = []

# for l in lambdas:
#    model = ...
#    train_errors.append(...)
#    test_errors.append(...)

# --- SECTION 4: Visualization ---
# TASK: Plot Training vs. Test Error.
# Invert the x-axis so complexity (low lambda) increases to the right.

# plt.figure(figsize=(10, 6))
# ...
# plt.gca().invert_xaxis()
# plt.show()