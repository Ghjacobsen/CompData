#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:42:38 2026

@author: sned
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DTU Colors for plotting
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def info_leakage_audit():
    '''
    Simulating pure noise to catch Data Leakage.
    '''
    np.random.seed(42)
    N, M = 50, 1000
    # Create pure random noise
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    print('--- Workflow A: The Crime (Leakage) ---')
    # 1. Standardize the WHOLE dataset (X)
    scaler_a = StandardScaler()
    X_scaled_all = scaler_a.fit_transform(X)
    
    # 2. Calculate absolute correlation between each feature and y (using ALL data)
    all_corrs = np.array([np.abs(np.corrcoef(X_scaled_all[:, i], y)[0, 1]) for i in range(M)])
    
    # 3. Select top 10 features
    top_indices_a = np.argsort(all_corrs)[-10:]
    corrs_a = np.sort(all_corrs[top_indices_a])[::-1]
    
    # 4. Create selected dataset and split
    X_selected_a = X_scaled_all[:, top_indices_a]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_selected_a, y, test_size=0.5, random_state=42
    )
    
    # 5. Fit and Score
    model_a = LinearRegression().fit(X_train_a, y_train_a)
    r2_a = model_a.score(X_test_a, y_test_a)
    print(f'Workflow A (Leaky) Test R^2: {r2_a:.3f}')

    print('\n--- Workflow B: The Audit (No Leakage) ---')
    # 1. Split FIRST
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    # 2. Standardize Training ONLY, then apply to Test
    scaler_b = StandardScaler().fit(X_train_raw)
    X_train_s = scaler_b.transform(X_train_raw)
    X_test_s = scaler_b.transform(X_test_raw)
    
    # 3. Calculate correlation using Training data ONLY
    train_corrs = np.array([np.abs(np.corrcoef(X_train_s[:, i], y_train)[0, 1]) for i in range(M)])
    
    # 4. Select top 10 features based on training correlations
    top_indices_b = np.argsort(train_corrs)[-10:]
    corrs_b_train = np.sort(train_corrs[top_indices_b])[::-1]
    
    # 5. Subsets
    X_train_sel = X_train_s[:, top_indices_b]
    X_test_sel = X_test_s[:, top_indices_b]
    
    # 6. Fit and Score
    model_b = LinearRegression().fit(X_train_sel, y_train)
    r2_b = model_b.score(X_test_sel, y_test)
    print(f'Workflow B (non-leaky) Test R^2: {r2_b:.3f}')

    # --- VISUALIZATION OF THE EVIDENCE ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 11), corrs_a, color=DTU_RED)
    plt.title('Leaky Correlations (Cheating)', color=DTU_NAVY, fontsize=14)
    plt.ylabel('Abs. Correlation with y')
    plt.xlabel('Top 10 Features (Whole Data)')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    # Note: We plot the correlations of these features on the TEST set for Workflow B
    # to show they don't actually hold up.
    test_corrs_b = np.array([np.abs(np.corrcoef(X_test_sel[:, i], y_test)[0, 1]) for i in range(10)])
    plt.bar(range(1, 11), test_corrs_b, color=DTU_NAVY)
    plt.title('non-leaky Correlations (Audited)', color=DTU_NAVY, fontsize=14)
    plt.ylabel('Abs. Correlation with y')
    plt.xlabel('Top 10 Features (Selected on Train)')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print('\nVERDICT:')
    print('Workflow A is a scientific crime because information from the test set')
    print('labels leaked into the feature selection step. By choosing features')
    print('that correlate with the labels across the entire dataset, we found')
    print('spurious patterns that happen to exist in the test set by chance.')
    print('Workflow B shows that when selection is properly isolated, these')
    print('patterns disappear on unseen data.')

if __name__ == '__main__':
    info_leakage_audit()