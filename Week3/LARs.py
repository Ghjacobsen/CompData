# lars_lasso_trace.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def plot_lars_lasso_comparison():
    # 1. Create a New "Divergence-Forcing" Dataset
    # We use a 'Redundant Leader' scenario.
    np.random.seed(10) # Different seed for a new data distribution
    n_samples, n_features = 25, 8
    
    # Base independent features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Feature 0 is a high-correlation 'proxy' for the signal in Feat 1 & 2
    # It will enter the model first because it has the highest initial correlation.
    X[:, 0] = (X[:, 1] + X[:, 2]) + np.random.normal(0, 0.1, n_samples)
    
    # The true response only depends on Features 1 and 2
    # Once 1 and 2 are both in, Feature 0 becomes a nuisance.
    y = 5 * X[:, 1] + 5 * X[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    # Standardize X (Mandatory for LARS equiangular geometry)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # 2. Compute the paths
    # LARS: Standard equiangular path (once in, stays in)
    # LASSO: LARS with the 'zero-crossing' drop rule
    _, _, coefs_lars = linear_model.lars_path(X, y, method='lars')
    _, _, coefs_lasso = linear_model.lars_path(X, y, method='lasso')

    # X-axis: Iteration Number (Step Index)
    xx_lars = np.arange(coefs_lars.shape[1])
    xx_lasso = np.arange(coefs_lasso.shape[1])

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    # --- LARS Plot ---
    for i in range(n_features):
        ax1.plot(xx_lars, coefs_lars[i], color=colors[i], label=f'Feature {i}', 
                 marker='o', markersize=5, linewidth=2)
    
    ax1.set_title(f'Pure LARS Path: {len(xx_lars)-1} Iterations', fontsize=14, color='blue')
    ax1.set_xlabel('Iteration (Step Number)', fontsize=12)
    ax1.set_ylabel('Coefficient Value', fontsize=12)
    ax1.set_xticks(xx_lars)
    ax1.grid(True, alpha=0.2)

    # --- LASSO Plot ---
    for i in range(n_features):
        ax2.plot(xx_lasso, coefs_lasso[i], color=colors[i], 
                 marker='o', markersize=5, linewidth=2)
    
    ax2.set_title(f'LASSO Path: {len(xx_lasso)-1} Iterations\n(Feature 0 is DROPPED at Step 8)', fontsize=14, color='red')
    ax2.set_xlabel('Iteration (Step Number)', fontsize=12)
    ax2.set_xticks(xx_lasso)
    ax2.grid(True, alpha=0.2)

    # Annotations for Feature 0 (the blue line usually)
    # In this dataset, Feature 0 is index 0.


    ax1.legend(loc='lower left', ncol=2, fontsize='small')
    plt.tight_layout()
    
    print(f"LARS Total Steps: {len(xx_lars)-1}")
    print(f"LASSO Total Steps: {len(xx_lasso)-1}")
    plt.show()

if __name__ == "__main__":
    plot_lars_lasso_comparison()