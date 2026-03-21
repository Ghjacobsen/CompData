#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:31:20 2026

@author: sned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def bootstrap_audit_solution():
    '''
    Full solution for quantifying feature reliability via Bootstrap using Ridge Regression.
    '''
    # Load Wine Quality data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(url, sep=';')
    except:
        print('Error: Could not retrieve data. Ensure internet connection is active.')
        return

    feature_names = df.drop('quality', axis=1).columns.tolist()
    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    y = df['quality'].values

    # Parameters
    # We use alpha=1.0 as the 'audited' complexity setting (from Exercise 2.2)
    alpha_val = 1.0
    B = 1000 
    N, M = X.shape
    
    boot_betas = np.zeros((B, M))

    print(f'--- Starting Ridge Bootstrap Audit (B={B}, alpha={alpha_val}) ---')

    # Perform Resampling
    for b in range(B):
        # Resample with replacement to simulate new experiments
        X_resamp, y_resamp = resample(X, y, replace=True, n_samples=N, random_state=b)
        
        # Fit Ridge model on bootstrap sample
        model = Ridge(alpha=alpha_val).fit(X_resamp, y_resamp)
        boot_betas[b, :] = model.coef_

    # Calculate 95% Percentile Confidence Intervals
    lower_bounds = np.percentile(boot_betas, 2.5, axis=0)
    upper_bounds = np.percentile(boot_betas, 97.5, axis=0)
    means = np.mean(boot_betas, axis=0)

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 8))
    
    # Calculate error lengths for plt.errorbar
    errors = np.array([means - lower_bounds, upper_bounds - means])
    
    # Plotting horizontal bars with DTU branding
    plt.errorbar(means, range(M), xerr=errors, fmt='o', 
                 color=DTU_RED, ecolor=DTU_NAVY, capsize=5, 
                 markersize=8, label='95% Bootstrap CI (Ridge)')
    
    # Add a vertical line at zero (The Auditor's Threshold)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.yticks(range(M), feature_names)
    plt.xlabel('Coefficient Value (Standardized)', fontsize=12, color=DTU_NAVY)
    plt.title('Bootstrap Audit: Ridge Feature Reliability', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- VERDICT ---
    print('\nAUDITOR VERDICT (Ridge alpha=1.0):')
    for i, name in enumerate(feature_names):
        if lower_bounds[i] <= 0 <= upper_bounds[i]:
            print(f'Reject: {name:20} - Interval {lower_bounds[i]:.3f} to {upper_bounds[i]:.3f} crosses zero.')
        else:
            print(f'Trust:  {name:20} - Interval {lower_bounds[i]:.3f} to {upper_bounds[i]:.3f} is stable.')

    print('\nScientific Note:')
    print('Unlike OLS, Ridge coefficients are shrunk toward zero.')
    print('The bootstrap confirms if a feature remains stable even under')
    print('regularization and data perturbation.')

if __name__ == '__main__':
    bootstrap_audit_solution()