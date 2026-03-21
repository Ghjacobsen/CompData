import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_animated_penalty(save_animation=False):
    # Create a grid of beta values for contour calculation
    res = 400
    limit = 2.0
    x = np.linspace(-limit, limit, res)
    y = np.linspace(-limit, limit, res)
    beta1, beta2 = np.meshgrid(x, y)

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    dtu_red = '#990000'
    
    # Static elements
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\beta_1$', fontsize=14)
    ax.set_ylabel(r'$\beta_2$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Pre-calculate static penalties for reference
    l1_static = np.abs(beta1) + np.abs(beta2)
    l2_static = np.sqrt(beta1**2 + beta2**2)
    
    # Draw static reference lines (faint)
    ax.contour(beta1, beta2, l2_static, levels=[1], colors='blue', linewidths=1, linestyles='dashed', alpha=0.4)
    ax.contour(beta1, beta2, l1_static, levels=[1], colors=dtu_red, linewidths=1, linestyles='dashed', alpha=0.4)

    # Initialize plot objects that will change
    contour_storage = {'collection': None}
    
    title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=15, fontweight='bold')
    desc_text = ax.text(0.5, 1.01, '', transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    def update(frame):
        alpha = frame
        
        # Clean up previous contours
        if contour_storage['collection'] is not None:
            for tp in contour_storage['collection'].collections:
                tp.remove()

        # Calculate Elastic Net Penalty: alpha*|B|_1 + (1-alpha)*|B|_2^2
        # Note: We use beta1^2 + beta2^2 for the L2 part of Elastic Net
        penalty = alpha * (np.abs(beta1) + np.abs(beta2)) + (1 - alpha) * (beta1**2 + beta2**2)
        
        # Determine color/text based on alpha
        if alpha <= 0.01:
            color, title = 'blue', "Ridge Regression Penalty ($L_2$)"
            desc = "Pure Ridge: No sparsity, grouping effect enabled."
        elif alpha >= 0.99:
            color, title = dtu_red, "LASSO Penalty ($L_1$)"
            desc = "Pure LASSO: Sharp corners on axes enable exact sparsity."
        else:
            color, title = 'orange', f"Elastic Net Penalty ($\\alpha = {alpha:.2f}$)"
            desc = "Hybrid: Combining sparsity (points) with grouping (rounded edges)."

        # Draw the new contour for the unit ball (level=1)
        contour_storage['collection'] = ax.contour(beta1, beta2, penalty, levels=[1], colors=color, linewidths=3)
        title_text.set_text(title)
        desc_text.set_text(desc)
        return []

    # Frames for a smooth transition
    alphas = np.concatenate([np.linspace(0, 1, 40), np.linspace(1, 0, 40)])
    
    # Static legend for references
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=1, ls='--', alpha=0.6, label='Reference: Ridge ($L_2$)'),
        Line2D([0], [0], color=dtu_red, lw=1, ls='--', alpha=0.6, label='Reference: LASSO ($L_1$)'),
        Line2D([0], [0], color='orange', lw=3, label='Active: Elastic Net')
    ]
    ax.legend(handles=custom_lines, loc='upper right')
    
    ani = FuncAnimation(fig, update, frames=alphas, interval=100, blit=False)
    
    if save_animation:
        print("Saving animation as penalty_transformation.gif...")
        ani.save('penalty_transformation.gif', writer='pillow', fps=10)
        print("Save complete.")
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set save_animation=True to export the gif
    plot_animated_penalty(save_animation=True)