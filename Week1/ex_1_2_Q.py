#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:45:31 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Learning Objective: Quantify the Bias-Variance tradeoff using Ridge lambda.

# --- SECTION 1: Parameters ---
np.random.seed(42)
n_samples = 100
n_simulations = 500
sigma = 1.0          # Noise level
beta_true = np.array([2, 0])
x_eval = np.array([[1, 1]])
target_val = (x_eval @ beta_true)[0]
lambdas = np.logspace(-2, 5, 25)

# Fixed X for consistent theoretical/empirical comparison
mean = [0, 0]
cov = [[1, 0.98], [0.98, 1]] # High correlation
X_fixed = np.random.multivariate_normal(mean, cov, n_samples)

# --- SECTION 2: Simulation ---
# TASK: Iterate through lambdas and simulations.
# 1. Generate y = X_fixed @ beta_true + noise.
# 2. Fit Ridge(alpha=l).
# 3. Predict at x_eval.
# 4. Calculate empirical Bias^2 and Variance.

bias_sq = []
variance = []

print('Simulating Ridge complexity sweep...')

for l in lambdas:
    preds = []
    for _ in range(n_simulations):
        noise = np.random.normal(0, sigma, n_samples)
        y = X_fixed @ beta_true + noise
        model = Ridge(alpha=l).fit(X_fixed, y)
        pred = model.predict(x_eval)[0]
        preds.append(pred)
    
    preds = np.array(preds)
    pred_mean = np.mean(preds)
    
    # Empirical Bias^2
    b_sq = (pred_mean - target_val) ** 2
    bias_sq.append(b_sq)
    
    # Empirical Variance
    var = np.var(preds)
    variance.append(var)

print(" Bias-Variance simulation complete.")
#print bias and variance averages
print(f"Average Bias^2: {np.mean(bias_sq)}")
print(f"Average Variance: {np.mean(variance)}")



# --- SECTION 3: Visualization ---
# TASK: Plot Bias^2, Variance, and Total (Bias^2 + Var) vs Lambdas.
# Remember to invert the x-axis for complexity.

plt.figure(figsize=(10, 6))
plt.plot(lambdas, bias_sq, label='Bias^2', color='blue')
plt.plot(lambdas, variance, label='Variance', color='orange')
plt.plot(lambdas, np.array(bias_sq) + np.array(variance), label='Total Error', color='green')
plt.xlabel('Ridge Lambda (Complexity)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff in Ridge Regression')
plt.legend()
plt.xscale('log')
plt.gca().invert_xaxis()
plt.show()