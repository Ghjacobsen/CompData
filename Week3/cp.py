# lars_lasso_trace.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def plot_lars_lasso_comparison():
    # 1. Create a "Divergence-Forcing" Dataset
    # We use a setup where Feature 0 is a decoy that gets dropped.
    np.random.seed(10) 
    n_samples, n_features = 40, 20
    
    # Generate Training Data
    X_train = np.random.normal(0, 1, (n_samples, n_features))
    X_train[:, 0] = (X_train[:, 1] + X_train[:, 2]) + np.random.normal(0, 0.1, n_samples)
    
    # Generate Test Data (to calculate true test error)
    X_test = np.random.normal(0, 1, (100, n_features))
    X_test[:, 0] = (X_test[:, 1] + X_test[:, 2]) + np.random.normal(0, 0.1, 100)
    
    # True signals: Features 1, 2, 3, 4
    true_beta = np.zeros(n_features)
    true_beta[1:5] = [5, 5, 3, 3]
    
    y_train = X_train @ true_beta + np.random.normal(0, 1, n_samples)
    y_test = X_test @ true_beta + np.random.normal(0, 1, 100)
    
    # Standardize based on training set
    mean_X = X_train.mean(axis=0)
    std_X = X_train.std(axis=0)
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    # 2. Compute the LASSO path (using LARS algorithm)
    # This gives us coefficients at every "event" (addition or deletion)
    alphas, active, coefs_lasso = linear_model.lars_path(X_train, y_train, method='lasso')

    # 3. Calculate metrics for each step
    # We estimate sigma^2 from the most complex model to calculate Cp
    final_res = y_train - X_train @ coefs_lasso[:, -1]
    sigma_sq = np.sum(final_res**2) / (n_samples - n_features)
    
    train_errors = []
    test_errors = []
    cp_values = []
    
    for k in range(coefs_lasso.shape[1]):
        beta_k = coefs_lasso[:, k]
        
        # Train MSE
        train_mse = np.mean((y_train - X_train @ beta_k)**2)
        train_errors.append(train_mse)
        
        # Test MSE
        test_mse = np.mean((y_test - X_test @ beta_k)**2)
        test_errors.append(test_mse)
        
        # Cp Statistic
        # Cp = RSS/sigma^2 - n + 2*df
        rss = np.sum((y_train - X_train @ beta_k)**2)
        df = k 
        cp = (rss / sigma_sq) - n_samples + 2 * df
        cp_values.append(cp)

    xx = np.arange(coefs_lasso.shape[1])
    best_cp_step = np.argmin(cp_values)
    best_test_step = np.argmin(test_errors)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, n_features))

    # --- TOP PLOT: Parameter Trace ---
    for i in range(n_features):
        ax1.plot(xx, coefs_lasso[i], color=colors[i], label=f'Feat {i}', 
                 marker='o', markersize=4, linewidth=2, alpha=0.8)
    
    ax1.axvline(best_cp_step, color='black', linestyle='--', linewidth=2, label='Min Cp')
    ax1.set_title('LASSO Parameter Trace (Iteration-based)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Coefficient Value', fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='x-small')

    # --- BOTTOM PLOT: Error Metrics ---
    # Plot Train/Test Error on Primary Y-axis
    ax2.plot(xx, train_errors, 'g-o', label='Train MSE', linewidth=2, markersize=5)
    ax2.plot(xx, test_errors, 'r-o', label='Test MSE', linewidth=2, markersize=5)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_xlabel('Iteration (Step Number / Model Complexity)', fontsize=12)
    
    # Plot Cp on Secondary Y-axis
    ax2_cp = ax2.twinx()
    ax2_cp.plot(xx, cp_values, 'b--D', label='Cp Statistic', linewidth=1, markersize=4, alpha=0.6)
    ax2_cp.set_ylabel('Cp Value', color='blue', fontsize=12)
    ax2_cp.tick_params(axis='y', labelcolor='blue')
    
    # Highlight the minima
    ax2.plot(best_test_step, test_errors[best_test_step], 'ro', markersize=12, fillstyle='none', markeredgewidth=2)
    ax2_cp.plot(best_cp_step, cp_values[best_cp_step], 'bs', markersize=12, fillstyle='none', markeredgewidth=2)

    # Combined Legend for the bottom plot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_cp.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    ax2.set_title('Model Assessment: Train/Test Error vs. $C_p$', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    
    plt.xticks(xx)
    plt.tight_layout()
    
    print(f"Optimal Model (Cp): Step {best_cp_step}")
    print(f"Optimal Model (Test): Step {best_test_step}")
    plt.show()

if __name__ == "__main__":
    plot_lars_lasso_comparison()