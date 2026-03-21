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

    print('--- Workflow A:Leakage ---')
    # TODO: Implement the 'Leaky' workflow
    # 1. Standardize the WHOLE dataset (X) using StandardScaler
    # 2. Calculate absolute correlation between each feature in X_scaled and y
    # 3. Select the indices of the top 10 features with highest correlation
    # 4. Create X_selected containing only these 10 features
    # 5. Split (X_selected, y) into 50/50 train and test sets
    # 6. Fit LinearRegression on training and print Test R^2
    
    # 1-2. Standardize full X, then compute abs correlations with y
    scaler_a = StandardScaler()
    X_scaled_a = scaler_a.fit_transform(X)
    corrs_full_a = np.abs(np.corrcoef(X_scaled_a, y, rowvar=False)[-1, :-1])
    # 3. Top 10 feature indices by correlation
    top10_idx_a = np.argsort(corrs_full_a)[-10:][::-1]
    # 4. Select top 10 features
    X_selected_a = X_scaled_a[:, top10_idx_a]
    # 5. Split into train/test
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_selected_a, y, test_size=0.5, random_state=42
    )
    # 6. Fit and score
    model_a = LinearRegression()
    model_a.fit(X_train_a, y_train_a)
    r2_a = model_a.score(X_test_a, y_test_a)
    # Placeholder for plot data (absolute correlations)
    corrs_a = corrs_full_a[top10_idx_a]
    print(f'Workflow A (Leaky) Test R^2: {r2_a:.3f}')

    print('\n--- Workflow B: The Audit (No Leakage) ---')
    # TODO: Implement the 'Honest' workflow
    # 1. Split the original (X, y) into 50/50 train and test sets FIRST
    # 2. Fit a StandardScaler on X_train only and transform both X_train and X_test
    # 3. Calculate correlation between X_train features and y_train ONLY
    # 4. Select the top 10 features based on these training correlations
    # 5. Create subsets of X_train and X_test using these indices
    # 6. Fit LinearRegression on training subset and print Test R^2
    
    # 1. Split before any preprocessing/selection
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    # 2. Fit scaler on training only, transform both
    scaler_b = StandardScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)
    # 3. Correlations on training only
    corrs_full_b = np.abs(np.corrcoef(X_train_b_scaled, y_train_b, rowvar=False)[-1, :-1])
    # 4. Top 10 by training correlations
    top10_idx_b = np.argsort(corrs_full_b)[-10:][::-1]
    # 5. Subset train/test
    X_train_b_sel = X_train_b_scaled[:, top10_idx_b]
    X_test_b_sel = X_test_b_scaled[:, top10_idx_b]
    # 6. Fit and score
    model_b = LinearRegression()
    model_b.fit(X_train_b_sel, y_train_b)
    r2_b = model_b.score(X_test_b_sel, y_test_b)
    # Placeholder for plot data (absolute correlations)
    corrs_b = corrs_full_b[top10_idx_b]
    print(f'Workflow B (non-leaky) Test R^2: {r2_b:.3f}')

    # --- VISUALIZATION OF THE EVIDENCE ---
    # TODO: Create a bar plot comparing the top 10 correlations for A and B
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), corrs_a, color=DTU_RED)
    plt.title('Leaky Correlations', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(range(10), corrs_b, color=DTU_NAVY)
    plt.title('Non-leaky Correlations (audited)', color=DTU_NAVY)
    plt.ylabel('Abs. Correlation with y')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print('\nVERDICT:')
    print('Workflow A leaks test information into feature selection, inflating correlations and R^2 on noise.')
    print('Workflow B keeps selection inside the training set, so the test R^2 collapses toward 0 or below.')
    print('Leaky selection is a scientific crime because it overstates performance and will not generalize.')

if __name__ == '__main__':
    info_leakage_audit()