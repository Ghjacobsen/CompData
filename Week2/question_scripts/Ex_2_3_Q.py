#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:52:59 2026

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

def knn_complexity_audit():
    '''
    Audit the complexity of KNN and apply the One-SE Rule.
    '''
    # Load Wine Data (simplified)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    y = df['quality'].values

    # Sweep through K neighbors
    k_values = np.arange(1, 101)
    K_folds = 10
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=42)
    
    cv_means = []
    cv_ses = []

    print('--- Auditing KNN Complexity ---')

    for k in k_values:
        fold_errors = []
        # TODO: Implement K-fold CV for the current k
        # 1. Loop through folds
        # 2. Fit KNeighborsRegressor(n_neighbors=k)
        # 3. Store MSE
        
        # TODO: Calculate Mean and SE
        pass

    # TODO: Find k_min and apply One-SE Rule
    # Note: Larger k = simpler model for KNN.
    
    # --- VISUALIZATION ---
    # TODO: Plot cv_means vs k_values with error bars
    # Add the One-SE threshold line

if __name__ == '__main__':
    knn_complexity_audit()