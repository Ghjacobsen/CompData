#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:30:03 2026

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

def bootstrap_audit_skeleton():
    '''
    Skeleton for quantifying feature reliability via Bootstrap resampling using Ridge.
    '''
    # 1. Load Wine Quality data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    feature_names = df.drop('quality', axis=1).columns.tolist()
    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    y = df['quality'].values

    # 2. Parameters
    # Slide 22: Use 1000-2000 replicates for confidence intervals
    B = 1000 
    N, M = X.shape
    
    # Selection: Use a fixed complexity (alpha) or the one selected in Exercise 2.2
    alpha_val = 1.0
    
    # Storage for bootstrap coefficients
    boot_betas = np.zeros((B, M))

    print(f'--- Starting Ridge Bootstrap Audit (B={B}, alpha={alpha_val}) ---')

    for b in range(B):
        # TODO: Implement Resampling
        # 1. Generate random indices with replacement (use resample from sklearn)
        # 2. Create X_resamp and y_resamp
        
        # TODO: Refit Ridge Model
        # 3. Fit Ridge(alpha=alpha_val) on the bootstrap sample
        # 4. Store coefficients (.coef_) in boot_betas[b, :]
        pass

    # 3. TODO: Calculate 95% Percentile Confidence Intervals
    # Hint: Use np.percentile(boot_betas, [2.5, 97.5], axis=0) to find 
    # the lower and upper bounds for each of the M features.
    
    # 4. TODO: Visualization
    # Create a horizontal error bar plot (plt.errorbar) showing the 
    # means and the 95% intervals for each feature.
    # Use DTU_RED for the markers and DTU_NAVY for the error bars.
    
    print('Audit Complete.')

if __name__ == '__main__':
    bootstrap_audit_skeleton()