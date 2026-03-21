#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:12:53 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model, preprocessing, metrics
from sklearn.model_selection import train_test_split

# --- 1. DATA PREPARATION ---
print('--- Phase 1: Loading and Standardizing Diabetes Data ---')
# 442 patients, 10 physiological predictors (age, bmi, bp, s1-s6)
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Standardize features (essential for penalties to be applied uniformly)
X = preprocessing.scale(X)
y = (y - np.mean(y)) / np.std(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. MODEL TRAINING ---
print('\n--- Phase 2: Training OLS, Ridge, LASSO, and Elastic Net ---')

results = {}

# A. Ordinary Least Squares (No Penalty)
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
results['OLS'] = ols

# B. Ridge Regression (L2 Penalty)
# Uses built-in Cross-Validation to find the best alpha (lambda)
ridge = linear_model.RidgeCV(alphas=np.logspace(-4, 2, 100))
ridge.fit(X_train, y_train)
results['Ridge'] = ridge

# C. LASSO Regression (L1 Penalty)
# Performs automatic feature selection
lasso = linear_model.LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
lasso.fit(X_train, y_train)
results['LASSO'] = lasso

# D. Elastic Net (Hybrid L1 + L2 Penalty)
# Finds balance between Sparsity (Lasso) and Grouping (Ridge)
en = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
                               alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
en.fit(X_train, y_train)
results['Elastic Net'] = en

# --- 3. EVALUATION ---
print('\n--- Phase 3: Model Performance Summary ---')
perf_data = []

for name, model in results.items():
    y_pred = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    # Count variables with non-zero coefficients
    active_features = np.sum(np.abs(model.coef_) > 1e-10)
    
    perf_data.append({
        'Model': name,
        'MSE': f'{mse:.4f}',
        'R2 Score': f'{r2:.4f}',
        'Active Features': f'{active_features}/{len(feature_names)}'
    })

print(pd.DataFrame(perf_data))

# --- 4. VISUALIZATION ---
plt.figure(figsize=(14, 8))
x_axis = np.arange(len(feature_names))
width = 0.2  # bar width

# Plotting coefficients for each method
plt.bar(x_axis - 1.5*width, results['OLS'].coef_, width, label='OLS (Full)', color='lightgrey')
plt.bar(x_axis - 0.5*width, results['Ridge'].coef_, width, label='Ridge (L2)', color='#005088')
plt.bar(x_axis + 0.5*width, results['LASSO'].coef_, width, label='LASSO (L1)', color='#990000')
plt.bar(x_axis + 1.5*width, results['Elastic Net'].coef_, width, label='Elastic Net', color='orange')

plt.axhline(0, color='black', linewidth=1, alpha=0.5)
plt.xticks(x_axis, feature_names, fontsize=11)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Comparison of Sparse and Dense Regression Coefficients', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Highlight specific features
plt.annotate('Sparsity: LASSO\nzeros out "age"', 
             xy=(0, 0), xytext=(0, -0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             bbox=dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9))

plt.tight_layout()
plt.show()

print('\n--- Theoretical Insights ---')
print(f"Optimal Alpha (Lambda) for LASSO: {lasso.alpha_:.4f}")
print(f"Optimal L1 Ratio (Alpha) for EN: {en.l1_ratio_}")
if en.l1_ratio_ == 1.0:
    print("Note: Elastic Net converged to pure LASSO (L1 ratio = 1.0).")