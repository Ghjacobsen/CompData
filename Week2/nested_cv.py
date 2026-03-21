#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:20:33 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def manual_nested_cv():
    '''
    Manual implementation of Nested CV to demonstrate Selection vs. Assessment.
    '''
    np.random.seed(42)
    # Generate data: 50 samples, 100 features (high complexity regime)
    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) # Pure noise to maximize selection bias

    lambdas = np.logspace(-2, 4, 20)
    
    # K-Fold Setup
    K_outer = 5
    K_inner = 5
    outer_kf = KFold(n_splits=K_outer, shuffle=True, random_state=42)
    inner_kf = KFold(n_splits=K_inner, shuffle=True, random_state=42)

    outer_test_errors = []
    inner_val_errors = [] # To track the 'optimistic' view

    print(f'--- Starting Nested CV Audit (Outer={K_outer}, Inner={K_inner}) ---')

    # --- OUTER LOOP: ASSESSMENT ---
    for i, (train_audit_idx, test_audit_idx) in enumerate(outer_kf.split(X)):
        X_train_audit, X_test_audit = X[train_audit_idx], X[test_audit_idx]
        y_train_audit, y_test_audit = y[train_audit_idx], y[test_audit_idx]
        
        # Scaling within the loop to prevent leakage
        scaler = StandardScaler().fit(X_train_audit)
        X_train_audit_s = scaler.transform(X_train_audit)
        X_test_audit_s = scaler.transform(X_test_audit)

        best_lambda = None
        min_inner_mse = float('inf')
        
        # --- INNER LOOP: SELECTION ---
        # We tune lambda on the current 'train_audit' data only
        for l in lambdas:
            fold_mses = []
            for train_inner_idx, val_inner_idx in inner_kf.split(X_train_audit_s):
                X_tr, X_val = X_train_audit_s[train_inner_idx], X_train_audit_s[val_inner_idx]
                y_tr, y_val = y_train_audit[train_inner_idx], y_train_audit[val_inner_idx]
                
                model = Ridge(alpha=l).fit(X_tr, y_tr)
                fold_mses.append(mean_squared_error(y_val, model.predict(X_val)))
            
            mean_inner_mse = np.mean(fold_mses)
            if mean_inner_mse < min_inner_mse:
                min_inner_mse = mean_inner_mse
                best_lambda = l
        
        # Track the 'Optimistic' error from the selection phase
        inner_val_errors.append(min_inner_mse)

        # Train final model for this fold using best_lambda and evaluate on Test Audit
        final_fold_model = Ridge(alpha=best_lambda).fit(X_train_audit_s, y_train_audit)
        test_mse = mean_squared_error(y_test_audit, final_fold_model.predict(X_test_audit_s))
        outer_test_errors.append(test_mse)
        
        print(f'Fold {i+1}: Best Lambda={best_lambda:.2f} | Inner MSE={min_inner_mse:.3f} | Outer MSE={test_mse:.3f}')

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    folds = np.arange(1, K_outer + 1)
    
    plt.plot(folds, inner_val_errors, 'o--', color=DTU_NAVY, label='Inner CV (Optimistic Selection)')
    plt.plot(folds, outer_test_errors, 's-', color=DTU_RED, label='Outer Test (Honest Assessment)')
    
    plt.fill_between(folds, inner_val_errors, outer_test_errors, color='gray', alpha=0.1, label='Selection-Induced Bias')
    
    plt.xlabel('Outer Fold Index', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Nested CV: Visualizing the "Optimism Gap"', fontsize=14, fontweight='bold', color=DTU_NAVY)
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print('\nFINAL VERDICT:')
    print(f'Mean Inner (Selection) Error: {np.mean(inner_val_errors):.3f}')
    print(f'Mean Outer (Assessment) Error: {np.mean(outer_test_errors):.3f}')
    print('The gap between these values is the cost of searching for the best model.')

if __name__ == '__main__':
    manual_nested_cv()