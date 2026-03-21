#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:10:18 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# DTU Aesthetics
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def exercise_2_3_analytical_guards():
    '''
    Skeleton for AIC vs BIC comparison.
    '''
    np.random.seed(42)
    N = 100
    M = 50
    X = np.random.randn(N, M)
    # Only the first 5 features are actually important
    y = X[:, :5] @ np.array([1, 2, 3, 4, 5]) + np.random.randn(N) * 2

    aics = []
    bics = []
    d_values = np.arange(1, M + 1)

    print('--- Running Exercise 2.3: AIC vs BIC ---')

    for d in d_values:
        # 1. Prepare data subset using the first 'd' features
        X_sub = X[:, :d]
        
        # 2. Fit OLS model
        # TODO: Fit LinearRegression()
        
        # 3. Calculate Residual Sum of Squares (RSS)
        # TODO: Calculate sum((y - y_pred)**2)
        
        # 4. Calculate Log-Likelihood term
        # Under Gaussian assumptions, -2*logL is proportional to N * ln(RSS/N)
        # TODO: logL_term = N * np.log(RSS / N)
        
        # 5. Compute AIC and BIC
        # TODO: AIC = logL_term + 2*d
        # TODO: BIC = logL_term + ln(N)*d
        pass

    # --- VISUALIZATION ---
    # TODO: Create a plot comparing AIC (DTU_RED) and BIC (DTU_NAVY)
    # Mark the minimum points for both criteria
    print('Plotting results...')

if __name__ == '__main__':
    exercise_2_3_analytical_guards()