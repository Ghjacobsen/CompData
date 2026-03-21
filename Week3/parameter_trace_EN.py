#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:27:21 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing

# --- 1. DATA PREPARATION ---
# Loading the Diabetes dataset: 442 patients, 10 predictors
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Standardize features (essential for path/trace visualization)
X = preprocessing.scale(X)
y = (y - np.mean(y)) / np.std(y)

# --- 2. COMPUTE ELASTIC NET PATHS ---
# Trace A: Near-LASSO (l1_ratio = 0.99)
# Trace B: Near-Ridge (l1_ratio = 0.10)
ratios = [0.99, 0.1]
titles = ['Trace A: Low Ridge Penalty (L1 ratio = 0.99)', 
          'Trace B: High Ridge Penalty (L1 ratio = 0.1)']
colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

for ax, l1_ratio, title in zip(axes, ratios, titles):
    # Compute the path: coefficients for a sequence of models
    # enet_path solves the model for a grid of alpha values (lambdas)
    alphas, coefs, _ = linear_model.enet_path(X, y, l1_ratio=l1_ratio, eps=1e-3)
    
    # We use the index of the alpha grid as the 'Iteration Number'
    # Iteration 0 corresponds to max regularization (all coefficients are zero)
    # Moving to the right (higher iteration) means regularization is decreasing
    iterations = np.arange(len(alphas))
    
    # Plotting each feature's coefficient path
    for i in range(len(feature_names)):
        ax.plot(iterations, coefs[i], color=colors[i], label=feature_names[i], linewidth=2)
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)

# Formatting the shared X-axis
axes[1].set_xlabel('Iteration Number (Path Index) → Decreasing Regularization', fontsize=12)
axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')

# Adding explanatory annotations
axes[0].annotate('Independent Entry:\nFeatures hit zero sharply\n(Lasso-like)', 
                 xy=(75, 0.2), xytext=(40, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

axes[1].annotate('Grouping Effect:\nFeatures co-evolve smoothly\n(Ridge-like)', 
                 xy=(75, 0.1), xytext=(40, 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

print('--- Interpretation for Iteration-based Traces ---')
print('Iteration 0: High regularization (all coefficients are zero).')
print('Final Iteration: Minimum regularization (approaching the OLS solution).')
print('Trace A (L1 Ratio 0.99): Features enter the model sharply and independently.')
print('Trace B (L1 Ratio 0.10): The L2 component stabilizes the paths, causing them to move together.')