import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import os

def plot_uncertainty_vs_dropout(uncertainties_dict, truth_dict, dropout_values, out_dir):
    """
    Plot mean uncertainties across different dropout rates for different truth classes.
    
    Parameters:
    - uncertainties_dict: Dictionary mapping dropout rate to uncertainty arrays
    - truth_dict: Dictionary mapping dropout rate to truth mask dictionaries
    - dropout_values: List of dropout values to plot
    - out_dir: Output directory for saving plots
    """
    # Sort dropout values for x-axis
    sorted_dropouts = sorted(dropout_values)
    
    # Define truth classes
    truth_classes = [
        ("target", "Target Truth", "tab:blue"),
        ("non_target", "Non-target Truth", "tab:green"), 
        ("false", "False", "tab:orange")
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For each truth class, plot a line
    for truth_class, truth_label, color in truth_classes:
        # Arrays to hold mean uncertainties for each dropout rate
        means = []
        errors = []
        
        # Calculate mean uncertainties for each dropout rate
        for dropout in sorted_dropouts:
            # Skip if this dropout doesn't exist
            if dropout not in uncertainties_dict or dropout not in truth_dict:
                means.append(np.nan)
                errors.append(np.nan)
                continue
                
            uncertainties = uncertainties_dict[dropout]
            truth_masks = truth_dict[dropout]
            
            # Apply truth class mask
            mask = truth_masks[truth_class]
            bin_uncertainties = uncertainties[mask]
            
            if len(bin_uncertainties) > 0:
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
            else:
                means.append(np.nan)
                errors.append(np.nan)
        
        # Convert to numpy arrays
        means = np.array(means)
        errors = np.array(errors)
        
        # Plot with error bands
        ax.plot(sorted_dropouts, means, '-', linewidth=2, color=color,
                label=f'{truth_label}', markersize=6)
        ax.fill_between(
            sorted_dropouts, 
            means - errors, 
            means + errors, 
            alpha=0.3,
            edgecolor=None,
            color=color
        )
    
    # Configure axis
    ax.set_xlabel('Dropout rate', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Mean standard deviation', fontsize=14, ha="right", y=0.95)
    
    # Set x-axis ticks to be exactly at our dropout values
    ax.set_xticks(sorted_dropouts)
    ax.set_xticklabels([f'{d:.2f}' for d in sorted_dropouts], rotation=45)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Apply ATLAS styling
    atlasify(" ",
        rf"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"MC Dropout with 100 forward passes" + "\n"
        + f"Trained on 1400 events" + "\n"
        + f"Evaluated on 50 events in valset"
    )
    
    fig.tight_layout()
    
    # Save the figure
    save_path = os.path.join(out_dir, "uncertainty_vs_dropout_rate.png")
    save_path_svg = os.path.join(out_dir, "uncertainty_vs_dropout_rate.svg")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Uncertainty vs dropout rate plot saved to {save_path}")


def plot_uncertainty_vs_dropout_heatmap(uncertainties_dict, scores_dict, truth_dict, dropout_values, out_dir):
    """
    Plot heatmaps of uncertainties vs. dropout rate and score for target, non-target, and false edges.
    
    Parameters:
    - uncertainties_dict: Dictionary mapping dropout rate to uncertainty arrays
    - scores_dict: Dictionary mapping dropout rate to score arrays
    - truth_dict: Dictionary mapping dropout rate to truth mask dictionaries
    - dropout_values: List of dropout values to plot
    - out_dir: Output directory for saving plots
    """
    # Sort dropout values for y-axis
    sorted_dropouts = sorted(dropout_values)
    
    # Create bins for dropout and scores
    dropout_bins = np.array(sorted_dropouts)
    score_bins = np.linspace(0, 1, 101)
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_key, suffix in [
        ("Target", "target", "target"),
        ("Non-target", "non_target", "non_target"),
        ("False", "false", "false")
    ]:
        
        # Collect all data for this edge type across all dropouts
        all_scores = []
        all_uncertainties = []
        all_dropouts = []
        
        for dropout in sorted_dropouts:
            if dropout not in uncertainties_dict or dropout not in scores_dict or dropout not in truth_dict:
                continue
                
            # Filter data for this edge type
            truth_mask = truth_dict[dropout][truth_key]
            edge_scores = scores_dict[dropout][truth_mask]
            edge_uncertainties = uncertainties_dict[dropout][truth_mask]
            
            if len(edge_scores) == 0:  # Skip if no edges of this type
                continue
                
            # Add dropout values for each edge
            edge_dropouts = np.full_like(edge_scores, dropout)
            
            all_scores.extend(edge_scores)
            all_uncertainties.extend(edge_uncertainties)
            all_dropouts.extend(edge_dropouts)
        
        if len(all_scores) == 0:  # Skip if no data collected
            continue
            
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_uncertainties = np.array(all_uncertainties)
        all_dropouts = np.array(all_dropouts)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute 2D histogram for counts
        H, xedges, yedges = np.histogram2d(
            all_scores,
            all_dropouts,
            bins=[score_bins, dropout_bins]
        )
        
        # Compute sum of uncertainties in each bin
        uncertainty_sum, _, _ = np.histogram2d(
            all_scores,
            all_dropouts,
            bins=[score_bins, dropout_bins],
            weights=all_uncertainties
        )
        
        # Calculate mean uncertainty per bin (avoid divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_uncertainty = np.where(H > 0, uncertainty_sum / H, np.nan)
        
        # Create heatmap
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        im = ax.pcolormesh(X, Y, mean_uncertainty, cmap='viridis', 
                          vmin=0, vmax=min(0.5, np.nanpercentile(mean_uncertainty, 99)))
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean Uncertainty (Std Dev)', fontsize=12)
        
        # Configure axis
        ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Dropout Rate', fontsize=14, ha="right", y=0.95)
        
        # Set y-axis ticks to be exactly at our dropout values
        ax.set_yticks(sorted_dropouts)
        ax.set_yticklabels([f'{d:.2f}' for d in sorted_dropouts])
        
        # Apply ATLAS styling
        atlasify(" ",
            rf"{'$p_T > 1$' if edge_type=='Target' else '$p_T > 0$'} GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with 100 forward passes" + "\n"
            + f"Trained on 1400 events" + "\n"
            + f"Evaluated on 50 events in valset" + "\n"
            + f"{edge_type} edges only", outside=True
        )
        
        fig.tight_layout()
        
        # Save the figure
        save_path = os.path.join(out_dir, f"uncertainty_vs_dropout_score_heatmap_{suffix}.png")
        save_path_svg = os.path.join(out_dir, f"uncertainty_vs_dropout_score_heatmap_{suffix}.svg")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path_svg, bbox_inches='tight')
        plt.close(fig)
        print(f"{edge_type} edges uncertainty vs. dropout/score heatmap saved to {save_path}")


# Define dropout values
dropout_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Prepare the data dictionaries
uncertainties_dict = {}
scores_dict = {}
truth_dict = {}

# Load data for each dropout value
for dropout in dropout_values:
    base_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/multi_drop/{dropout}/"
    
    uncertainty_file = base_path + "all_flat_uncertainties_1400.txt"
    scores_file = base_path + "all_flat_scores_1400.txt"
    target_truth_file = base_path + "all_flat_target_truth_1400.txt"
    non_target_truth_file = base_path + "all_flat_non_target_truth_1400.txt"
    false_file = base_path + "all_flat_false_1400.txt"
    
    try:
        # Load uncertainty and score data
        uncertainties_dict[dropout] = np.loadtxt(uncertainty_file)
        scores_dict[dropout] = np.loadtxt(scores_file)
        
        # Load truth masks explicitly
        target_truth = np.loadtxt(target_truth_file).astype(bool)
        non_target_truth = np.loadtxt(non_target_truth_file).astype(bool)
        false_truth = np.loadtxt(false_file).astype(bool)
        
        truth_dict[dropout] = {
            "target": target_truth,
            "non_target": non_target_truth,
            "false": false_truth
        }
        
        print(f"Loaded data for dropout rate {dropout}")
        
    except FileNotFoundError as e:
        print(f"Warning: Could not load data for dropout rate {dropout} - {e}")
        continue

# Set output directory
out_dir = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/dropout_comparison/"
os.makedirs(out_dir, exist_ok=True)

# Generate the uncertainty vs dropout plot
if uncertainties_dict and truth_dict:
    plot_uncertainty_vs_dropout(uncertainties_dict, truth_dict, 
                               list(uncertainties_dict.keys()), out_dir)
else:
    print("No data loaded for line plot. Please check file paths.")

# Generate the uncertainty vs dropout heatmaps
if uncertainties_dict and scores_dict and truth_dict:
    plot_uncertainty_vs_dropout_heatmap(uncertainties_dict, scores_dict, truth_dict,
                                       list(uncertainties_dict.keys()), out_dir)
else:
    print("No data loaded for heatmap. Please check file paths.")