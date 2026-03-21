#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:43:08 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Learning Objective: Observe coefficient instability and its link to collinearity.

# --- SECTION 1: Parameters ---
np.random.seed(42) 
n_samples = 1000
n_simulations = 500
sigma = 1.0
rho = 0.98
beta_true = np.array([2, 0])
x_test = np.array([[1, 1]])
target_val = (x_test @ beta_true)[0]

def generate_data(n, rho, sigma):
    '''
    TASK: Generate synthetic data.
    1. Create X with two features correlated by rho (use multivariate_normal).
    2. Generate y = X @ beta_true + noise.
    '''
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    X = np.random.multivariate_normal(mean, cov, n)
    noise = np.random.normal(0, sigma, n)
    y = X @ beta_true + noise
    return X, y

# --- SECTION 2: Simulation ---
# TASK: Run a loop for n_simulations.
all_betas = []
all_preds = []

print('Running simulations...')
for _ in range(n_simulations):
    X, y = generate_data(n_samples, rho, sigma)
    model = LinearRegression().fit(X, y)
    all_betas.append(model.coef_)
    all_preds.append(model.predict(x_test)[0])

print('Simulations complete.')


# --- SECTION 3: Calculations ---
# TASK: Calculate the following metrics:
# 1. Mean and Variance of the estimated coefficients (betas).
# 2. The Bias^2 at x_test.
# 3. The Variance at x_test.

beta_mean = np.mean(all_betas, axis=0)
beta_variance = np.var(all_betas, axis=0)

bias_sq = (np.mean(all_preds) - target_val)**2

variance = np.var(all_preds)

# --- SECTION 4: Visualization ---
# TASK: Create a histogram of estimated Beta 1 and Beta 2.

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist([b[0] for b in all_betas], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Estimated Beta 1')
plt.xlabel('Beta 1 Estimates')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist([b[1] for b in all_betas], bins=30, color='salmon', edgecolor='black')
plt.title('Histogram of Estimated Beta 2')
plt.xlabel('Beta 2 Estimates')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Function to simulate OLS instability

def simulate_ols_instability(n=100, perturbation=0.01):
    np.random.seed(0)
    X = np.random.rand(n, 1) * 10  # Feature
    true_beta = 2.0
    y = true_beta * X.flatten() + np.random.normal(0, 1, n)  # Target variable

    # Fit OLS model without perturbation
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    original_beta = model.params

    # Introduce perturbation
    X_perturbed = X + np.random.normal(0, perturbation, X.shape)
    X_perturbed_with_const = sm.add_constant(X_perturbed)
    model_perturbed = sm.OLS(y, X_perturbed_with_const).fit()
    perturbed_beta = model_perturbed.params

    return original_beta, perturbed_beta

# Run simulation
original_beta, perturbed_beta = simulate_ols_instability()
print(f'Original Beta: {original_beta}, Perturbed Beta: {perturbed_beta}')

# Function to explore identifiability with collinearity

def explore_collinearity(n=100, rho=0.9):
    np.random.seed(0)
    # Generate correlated features
    X1 = np.random.rand(n)
    X2 = rho * X1 + (1 - rho) * np.random.rand(n)  # Correlated feature
    y = 2 * X1 + 3 * X2 + np.random.normal(0, 1, n)

    X = np.column_stack((X1, X2))
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model.params, model.bse  # Return coefficients and standard errors

# Run collinearity exploration
params, bse = explore_collinearity()
print(f'Parameters: {params}, Standard Errors: {bse}')
