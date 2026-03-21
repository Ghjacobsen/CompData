#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:14:35 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def analyze_wine_ridge_selection():
    # Load the Wine dataset
    data = load_wine()
    X_full = data.data
    # Standardize features (crucial for Ridge)
    X_full = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0)
    y = data.target 
    
    N, p = X_full.shape
    
    # Define a range of lambda (alpha) values
    # We use a log scale to cover a wide range of regularization
    alphas = np.logspace(-3, 5, 50)
    
    train_errors = []
    cv_scores = []
    aic_scores = []
    bic_scores = []
    dof_values = []
    
    # 1. Estimate sigma^2 from the OLS model (alpha=0)
    ols_model = Ridge(alpha=1e-10).fit(X_full, y) # tiny alpha to approximate OLS
    y_pred_ols = ols_model.predict(X_full)
    sigma_sq_est = np.sum((y - y_pred_ols)**2) / (N - p - 1)

    # Precompute X^T X for effective degrees of freedom calculation
    # df(lambda) = trace(X(X'X + lambda*I)^-1 X') = trace((X'X + lambda*I)^-1 X'X)
    XtX = X_full.T @ X_full
    eigenvalues = np.linalg.eigvalsh(XtX)

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_full, y)
        
        # Training Error (MSE)
        y_pred = model.predict(X_full)
        mse = mean_squared_error(y, y_pred)
        train_errors.append(mse)
        
        # CV Error
        cv_mse = -np.mean(cross_val_score(model, X_full, y, scoring='neg_mean_squared_error', cv=5))
        cv_scores.append(cv_mse)
        
        # 2. Calculate Effective Degrees of Freedom
        # df = sum(eigenvalues / (eigenvalues + alpha))
        df_lambda = np.sum(eigenvalues / (eigenvalues + alpha))
        dof_values.append(df_lambda)
        
        # 3. AIC: MSE + 2 * (df/N) * sigma_sq
        aic = mse + (2 * df_lambda * sigma_sq_est) / N
        aic_scores.append(aic)
        
        # 4. BIC: MSE + (log(N) * df/N) * sigma_sq
        bic = mse + (np.log(N) * df_lambda * sigma_sq_est) / N
        bic_scores.append(bic)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.style.use('seaborn-v0_8-muted')

    # Top Plot: Selection Criteria vs Lambda
    ax1.plot(alphas, train_errors, 'k--', alpha=0.4, label='Training Error')
    ax1.plot(alphas, cv_scores, 'b-', label='5-Fold CV Error', linewidth=2)
    ax1.plot(alphas, aic_scores, 'g-', label='AIC / $C_p$', linewidth=2)
    ax1.plot(alphas, bic_scores, 'r-', label='BIC', linewidth=2)
    
    # Identify the best models (minimums)
    best_cv_alpha = alphas[np.argmin(cv_scores)]
    best_bic_alpha = alphas[np.argmin(bic_scores)]
    
    ax1.axvline(best_cv_alpha, color='blue', linestyle='--', alpha=0.5, label=f'Best CV ($\lambda \approx {best_cv_alpha:.1f}$)')
    ax1.axvline(best_bic_alpha, color='red', linestyle='--', alpha=0.5, label=f'Best BIC ($\lambda \approx {best_bic_alpha:.1f}$)')
    
    ax1.set_xscale('log')
    ax1.set_ylabel('Criterion Value / MSE', fontsize=12)
    ax1.set_title(f'Ridge Regression Model Selection (N={N})', fontsize=14)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Bottom Plot: Effective Degrees of Freedom vs Lambda
    ax2.plot(alphas, dof_values, color='purple', linewidth=2.5, label='Effective DoF $df(\lambda)$')
    ax2.fill_between(alphas, dof_values, alpha=0.2, color='purple')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Parameter $\lambda$ (log scale)', fontsize=12)
    ax2.set_ylabel('Degrees of Freedom', fontsize=12)
    ax2.set_title('Effective Model Complexity vs. $\lambda$', fontsize=14)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_wine_ridge_selection()