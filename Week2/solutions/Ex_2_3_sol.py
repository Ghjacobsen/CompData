#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:53:39 2026

@author: sned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def knn_complexity_audit_solution():
    '''
    Full solution for KNN complexity audit and One-SE Rule.
    '''
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    y = df['quality'].values

    k_values = np.arange(1, 300)
    K_folds = 10
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=42)
    
    cv_means = []
    cv_ses = []

    print('--- Auditing KNN Complexity ---')

    for k in k_values:
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = KNeighborsRegressor(n_neighbors=k).fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_errors.append(mean_squared_error(y_val, preds))
        
        cv_means.append(np.mean(fold_errors))
        cv_ses.append(np.std(fold_errors) / np.sqrt(K_folds))
        
        print(k)

    cv_means = np.array(cv_means)
    cv_ses = np.array(cv_ses)

    # 1. Find best k (minimum error)
    idx_min = np.argmin(cv_means)
    k_min = k_values[idx_min]
    err_min = cv_means[idx_min]
    
    # 2. Apply One-SE Rule
    threshold = err_min + cv_ses[idx_min]
    # For KNN, LARGER k means a simpler model (more averaging)
    # So we look for the largest k that is still under the threshold
    possible_k_idx = np.where(cv_means <= threshold)[0]
    idx_1se = np.max(possible_k_idx)
    k_1se = k_values[idx_1se]

    print(f'Optimal k (Min Error): {k_min}')
    print(f'Audited k (One-SE Rule): {k_1se}')

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, cv_means, yerr=cv_ses, fmt='-', 
                 color=DTU_RED, ecolor='lightgray', alpha=0.7, label='CV MSE')
    
    plt.axhline(threshold, color=DTU_NAVY, linestyle='--', label='One-SE Threshold')
    plt.scatter(k_min, err_min, color=DTU_NAVY, zorder=5, label=f'Min k={k_min}')
    plt.scatter(k_1se, cv_means[idx_1se], color='black', marker='x', s=100, zorder=5, label=f'1-SE k={k_1se}')

    plt.xlabel('Number of Neighbors (k)', color=DTU_NAVY, fontsize=12)
    plt.ylabel('Mean Squared Error', color=DTU_NAVY, fontsize=12)
    plt.title('KNN Complexity Audit: Bias-Variance in Action', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print('\nVERDICT:')
    print(f'The auditor chooses k = {k_1se}.')
    print('Small k leads to high-variance "jagged" predictions.')
    print('By choosing the larger k within the SE threshold, we prefer')
    print('a smoother, more robust model that generalizes better.')

if __name__ == '__main__':
    knn_complexity_audit_solution()