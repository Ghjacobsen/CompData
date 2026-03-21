#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:29:49 2026

@author: sned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def wine_audit_one_se():
    '''
    Implementing the One-SE rule on the Wine Quality dataset.
    '''
    # Load data (Ensure the file is in your directory)
    # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    # df = pd.read_csv(url, sep=';')
    
    # For now, we assume data is loaded and split into X and y
    # X = df.drop('quality', axis=1).values
    # y = df['quality'].values

    # Complexity sweep
    lambdas = np.logspace(-3, 5, 20)
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    cv_means = []
    cv_ses = []

    print('--- Starting 10-Fold Cross-Validation ---')
    
    for l in lambdas:
        fold_errors = []
        # TODO: Loop through kf.split(X)
        # 1. Fit Ridge model with alpha=l on training folds
        # 2. Calculate MSE on the validation fold
        # 3. Append to fold_errors
        
        # TODO: Calculate Mean and SE for this lambda
        # Mean = np.mean(fold_errors)
        # SE = np.std(fold_errors) / np.sqrt(K)
        pass

    # TODO: Identify lambda_min (index of the lowest cv_mean)
    
    # TODO: Apply the One-SE Rule
    # 1. threshold = cv_means[idx_min] + cv_ses[idx_min]
    # 2. Find the largest lambda where cv_mean <= threshold
    
    # --- VISUALIZATION ---
    # TODO: Create the CV Error Plot
    # Use plt.errorbar(lambdas, cv_means, yerr=cv_ses, color=DTU_RED)
    # Add a horizontal line for the threshold using DTU_NAVY
    # Set x-axis to log scale
    
    print('Audit Complete.')

if __name__ == '__main__':
    wine_audit_one_se()