#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:31:27 2026

@author: sned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def wine_audit_one_se_solution():
    '''
    Full solution for Ridge CV and the One-SE Rule.
    '''
    # Load Wine Quality dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(url, sep=';')
    except:
        print('Error loading data. Please check internet connection.')
        return

    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    # Scaling is essential for Ridge
    X = StandardScaler().fit_transform(X)

    lambdas = np.logspace(-2, 6, 25)
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    cv_means = []
    cv_ses = []

    print('--- Auditing Model Path ---')
    
    for l in lambdas:
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = Ridge(alpha=l).fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_errors.append(mean_squared_error(y_val, preds))
        
        cv_means.append(np.mean(fold_errors))
        cv_ses.append(np.std(fold_errors) / np.sqrt(K))

    cv_means = np.array(cv_means)
    cv_ses = np.array(cv_ses)

    # 1. Find lambda_min
    idx_min = np.argmin(cv_means)
    l_min = lambdas[idx_min]
    err_min = cv_means[idx_min]
    se_min = cv_ses[idx_min]

    # 2. Apply One-SE Rule
    threshold = err_min + se_min
    # We want the largest lambda (simplest model) that is below the threshold
    # Note: For Ridge, larger alpha = simpler model
    possible_lambdas_idx = np.where(cv_means <= threshold)[0]
    idx_1se = np.max(possible_lambdas_idx)
    l_1se = lambdas[idx_1se]

    print(f'Lambda Min: {l_min:.2f} (Error: {err_min:.4f})')
    print(f'Lambda 1-SE: {l_1se:.2f} (Error: {cv_means[idx_1se]:.4f})')

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(lambdas, cv_means, yerr=cv_ses, fmt='o-', 
                 color=DTU_RED, ecolor='lightgray', capsize=3, label='CV Mean Error')
    
    plt.axhline(threshold, color=DTU_NAVY, linestyle='--', label='One-SE Threshold')
    plt.axvline(l_min, color='gray', linestyle=':', label='Min Lambda')
    plt.axvline(l_1se, color=DTU_NAVY, linestyle='-', label='Selected (One-SE)')

    plt.xscale('log')
    plt.xlabel('Complexity Penalty (Lambda)', color=DTU_NAVY, fontsize=12)
    plt.ylabel('Mean Squared Error', color=DTU_NAVY, fontsize=12)
    plt.title('Wine Quality Audit: The One-SE Rule', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print('\nVERDICT:')
    print(f'The auditor selects lambda = {l_1se:.2f}.')
    print('While its error is slightly higher than the minimum, it is within')
    print('one standard error, meaning the difference is likely noise.')
    print('The larger lambda results in a more parsimonious, robust model.')

if __name__ == '__main__':
    wine_audit_one_se_solution()