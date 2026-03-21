#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:14:45 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# DTU Aesthetics
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def exercise_2_3_analytical_guards_sol():
    '''
    Full solution for AIC vs BIC comparison.
    '''
    np.random.seed(42)
    N = 100
    M = 50
    X = np.random.randn(N, M)
    # Only the first 5 features contain signal, the rest are noise
    y = X[:, :5] @ np.array([1, 2, 3, 4, 5]) + np.random.randn(N) * 2

    aics = []
    bics = []
    d_values = np.arange(1, M + 1)

    print('--- Running Exercise 2.3: AIC vs BIC ---')

    for d in d_values:
        # 1. Fit OLS using first 'd' features
        X_sub = X[:, :d]
        model = LinearRegression().fit(X_sub, y)
        preds = model.predict(X_sub)
        
        # 2. Residual Sum of Squares
        rss = np.sum((y - preds)**2)
        
        # 3. Log-Likelihood term (assuming Gaussian noise)
        # -2 * logL is proportional to N * ln(RSS/N)
        logL_penalty = N * np.log(rss / N)
        
        # 4. AIC = -2*logL + 2*d
        aics.append(logL_penalty + 2 * d)
        
        # 5. BIC = -2*logL + ln(N)*d
        bics.append(logL_penalty + np.log(N) * d)

    aics = np.array(aics)
    bics = np.array(bics)
    
    # Identify optimal points
    d_aic = d_values[np.argmin(aics)]
    d_bic = d_values[np.argmin(bics)]
    
    print(f'Optimal model size (AIC): {d_aic}')
    print(f'Optimal model size (BIC): {d_bic}')

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, aics, 'o-', color=DTU_RED, label='AIC (-2L + 2d)', markersize=4)
    plt.plot(d_values, bics, 's-', color=DTU_NAVY, label=f'BIC (-2L + ln({N})d)', markersize=4)
    
    # Highlight minimums
    plt.axvline(d_aic, color=DTU_RED, linestyle='--', alpha=0.6, label=f'Best d (AIC)={d_aic}')
    plt.axvline(d_bic, color=DTU_NAVY, linestyle='--', alpha=0.6, label=f'Best d (BIC)={d_bic}')
    
    plt.title('Analytical Model Selection: AIC vs BIC', color=DTU_NAVY, fontweight='bold', fontsize=14)
    plt.xlabel('Number of Features (d)', fontsize=12)
    plt.ylabel('Criterion Value (Lower is better)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print('\nVERDICT:')
    print('BIC uses a heavier penalty (ln(N) vs 2). Because N=100, ln(N) is approx 4.6,')
    print('making BIC a stricter auditor that favors simpler models than AIC.')

if __name__ == '__main__':
    exercise_2_3_analytical_guards_sol()