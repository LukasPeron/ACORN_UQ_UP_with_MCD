import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import os
from matplotlib.lines import Line2D

def plot_uncertainty_comparison(scores_dict, uncertainties_dicts, uncertainty_types, truth_dict, config, out_dir):
    """
    Plot mean uncertainties across different train event counts for various score bins,
    with separate figures for true and false edges.
    
    Parameters:
    - scores_dict: Dictionary mapping n_event to score arrays
    - uncertainties_dicts: Dictionary mapping uncertainty type to dictionaries of n_event to uncertainty arrays
    - uncertainty_types: List of uncertainty types to plot
    - truth_dict: Dictionary mapping n_event to truth arrays
    - config: Configuration dictionary
    - out_dir: Output directory for saving plots
    """
    # Define 10 score bins for better granularity
    score_bins = np.linspace(0, 1, 11)
    bin_labels = [f"{score_bins[i]:.1f}-{score_bins[i+1]:.1f}" for i in range(len(score_bins)-1)]
    
    # Sort event counts for x-axis
    sorted_events = sorted(scores_dict.keys())
    max_event_count = max(sorted_events)  # This should be 1400
    
    # Create separate figures for each uncertainty type
    for uncertainty_type in uncertainty_types:
        uncertainties_dict = uncertainties_dicts[uncertainty_type]
        
        # Create two separate figures - one for true edges, one for false edges
        for edge_type in ["true", "false"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # For each score bin, calculate uncertainties across event counts
            for bin_idx in range(len(score_bins)-1):
                low_score, high_score = score_bins[bin_idx], score_bins[bin_idx+1]
                
                # Arrays to hold mean uncertainties for each event count
                means = []
                errors = []
                
                # Calculate mean uncertainties for each event count
                for n_event in sorted_events:
                    scores = scores_dict[n_event]
                    uncertainties = uncertainties_dict[n_event]
                    truth = truth_dict[n_event]
                    
                    if edge_type == "true":
                        # True edges
                        mask = truth.astype(bool)
                    else:
                        # False edges
                        mask = ~truth.astype(bool)
                        
                    score_bin_mask = (scores >= low_score) & (scores < high_score)
                    bin_mask = mask & score_bin_mask
                    
                    if np.sum(bin_mask) > 10:
                        bin_uncertainties = uncertainties[bin_mask]
                        means.append(np.mean(bin_uncertainties))
                        errors.append(np.std(bin_uncertainties) / np.sqrt(np.sum(bin_mask)))
                    else:
                        means.append(np.nan)
                        errors.append(np.nan)
                
                # Convert to numpy arrays
                means = np.array(means)
                errors = np.array(errors)
                
                # Define colors using viridis colormap
                cmap = plt.cm.viridis
                colors = [cmap(i/(len(score_bins)-1)) for i in range(len(score_bins)-1)]
                
                # Plot with error bands
                ax.plot(sorted_events, means, '-o', color=colors[bin_idx], linewidth=2, 
                        label=f'Score [{bin_labels[bin_idx]})')
                ax.fill_between(
                    sorted_events, 
                    means - errors, 
                    means + errors, 
                    alpha=0.3,
                    color=colors[bin_idx],
                    edgecolor=None
                )
            
            # Add horizontal lines for aleatoric uncertainty for Total uncertainty plots
            if uncertainty_type == "Total":
                aleatoric_dict = uncertainties_dicts["Aleatoric"]
                
                for bin_idx in range(len(score_bins)-1):
                    low_score, high_score = score_bins[bin_idx], score_bins[bin_idx+1]
                    
                    # Get aleatoric uncertainty for this bin at the maximum event count
                    scores = scores_dict[max_event_count]
                    uncertainties = aleatoric_dict[max_event_count]
                    truth = truth_dict[max_event_count]
                    
                    if edge_type == "true":
                        mask = truth.astype(bool)
                    else:
                        mask = ~truth.astype(bool)
                        
                    score_bin_mask = (scores >= low_score) & (scores < high_score)
                    bin_mask = mask & score_bin_mask
                    
                    if np.sum(bin_mask) > 10:
                        bin_uncertainties = uncertainties[bin_mask]
                        aleatoric_value = np.mean(bin_uncertainties)
                        
                        # Draw horizontal line with the same color
                        ax.axhline(
                            y=aleatoric_value, 
                            color=colors[bin_idx], 
                            linestyle='--', 
                            alpha=0.7,
                            linewidth=1.5
                        )
            
            # Configure axis
            ax.set_xlabel('Number of Training Events', fontsize=14, ha="right", x=0.95)
            ax.set_ylabel(f'{uncertainty_type} Uncertainty', fontsize=14, ha="right", y=0.95)
            ax.set_ylim(0, 0.4)
            
            # Set x-axis ticks to be exactly at our event counts
            ax.set_xticks(sorted_events)
            ax.set_xticklabels(sorted_events)
            
            # Add legend with sorted score bins
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_order = sorted(range(len(labels)), key=lambda i: float(labels[i].split('[')[1].split('-')[0]))
                handles = [handles[i] for i in legend_order]
                labels = [labels[i] for i in legend_order]
                ax.legend(handles, labels, loc='upper right', fontsize=10, ncol=2)
            
            # Apply ATLAS styling
            additional_text = ""
            if uncertainty_type == "Total":
                additional_text = " - Dashed lines show aleatoric uncertainty at 1400 events"
                
            atlasify(" ",
                rf"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {config.get('dropout', 0.0)}"
                + "\n"
                + f"Evaluated on 50 events in valset" + "\n"
                f"{'True' if edge_type == 'true' else 'False'} edges only - {uncertainty_type}"
                + additional_text,
            )
            
            fig.tight_layout()
            
            # Save the figure
            save_path = os.path.join(out_dir, f"{uncertainty_type}_vs_events_{edge_type}_edges.png")
            save_path_svg = os.path.join(out_dir, f"{uncertainty_type}_vs_events_{edge_type}_edges.svg")
            fig.savefig(save_path)
            fig.savefig(save_path_svg)
            plt.close(fig)
            print(f"{edge_type.capitalize()} edges plot for {uncertainty_type} saved to {save_path}")

def plot_overall_uncertainty_comparison(scores_dict, uncertainties_dicts, uncertainty_types, truth_dict, config, out_dir):
    """
    Plot mean uncertainties across different train event counts WITHOUT score binning,
    with separate figures for true and false edges.
    
    Parameters:
    - scores_dict: Dictionary mapping n_event to score arrays
    - uncertainties_dicts: Dictionary mapping uncertainty type to dictionaries of n_event to uncertainty arrays
    - uncertainty_types: List of uncertainty types to plot
    - truth_dict: Dictionary mapping n_event to truth arrays
    - config: Configuration dictionary
    - out_dir: Output directory for saving plots
    """
    # Sort event counts for x-axis
    sorted_events = sorted(scores_dict.keys())
    
    # Create separate figures for true and false edges
    for yscale in ["linear", "log"]:
        for edge_type in ["true", "false"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # For each uncertainty type, plot a line
            for uncertainty_type in uncertainty_types:
                uncertainties_dict = uncertainties_dicts[uncertainty_type]
                
                # Arrays to hold mean uncertainties for each event count
                means = []
                errors = []
                
                # Calculate mean uncertainties for each event count
                for n_event in sorted_events:
                    uncertainties = uncertainties_dict[n_event]
                    truth = truth_dict[n_event]
                    
                    if edge_type == "true":
                        # True edges
                        mask = truth.astype(bool)
                    else:
                        # False edges
                        mask = ~truth.astype(bool)
                        
                    bin_uncertainties = uncertainties[mask]
                    means.append(np.mean(bin_uncertainties))
                    errors.append(np.std(bin_uncertainties) / np.sqrt(np.sum(mask)))
                
                # Convert to numpy arrays
                means = np.array(means)
                errors = np.array(errors)
                
                # Define colors for different uncertainty types
                colors = {
                    "Standard deviation": "blue", 
                    "Total": "red", 
                    "Epistemic": "green", 
                    "Aleatoric": "purple"
                }
                
                # Plot with error bands
                ax.plot(sorted_events, means, '-o', color=colors[uncertainty_type], linewidth=2, 
                        label=f'{uncertainty_type}')
                ax.fill_between(
                    sorted_events, 
                    means - errors, 
                    means + errors, 
                    alpha=0.3,
                    color=colors[uncertainty_type],
                    edgecolor=None
                )
            
            # Configure axis
            ax.set_xlabel('Number of Training Events', fontsize=14, ha="right", x=0.95)
            ax.set_ylabel('Uncertainty', fontsize=14, ha="right", y=0.95)
            
            # Set x-axis ticks to be exactly at our event counts
            ax.set_xticks(sorted_events)
            ax.set_xticklabels(sorted_events)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=12)
            ax.set_yscale(yscale)
            ax.set_ylim(0, 0.4)
            
            # Apply ATLAS styling
            atlasify(" ",
                rf"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {config.get('dropout', 0.0)}"
                + "\n"
                + f"Evaluated on 50 events in valset" + "\n"
                f"{'True' if edge_type == 'true' else 'False'} edges - All uncertainties"
            )
            
            fig.tight_layout()
            
            # Save the figure
            save_path = os.path.join(out_dir, f"all_uncertainties_vs_events_{edge_type}_edges_{yscale}.png")
            save_path_svg = os.path.join(out_dir, f"all_uncertainties_vs_events_{edge_type}_edges_{yscale}.svg")
            fig.savefig(save_path)
            fig.savefig(save_path_svg)
            plt.close(fig)
            print(f"{edge_type.capitalize()} edges overall uncertainty plot saved to {save_path}")

hardcuts = False  # Set to True for hard cuts
if hardcuts:
    n_events = [0, 50, 100, 200, 400, 800, 1400]
else:
    n_events = [100, 200, 400, 800, 1400]
prefix = ["eta_", "pt_", "scores_", "scores_track_", "target_truth_", "non_target_truth_", "uncertainties_"]
all_prefix = "all_flat_"

# Prepare the data dictionaries
scores_dict = {}
uncertainties_dict = {}  # Standard deviation uncertainties
total_uncertainties_dict = {}  # Total uncertainties
epistemic_uncertainties_dict = {}  # Epistemic uncertainties
aleatoric_uncertainties_dict = {}  # Aleatoric uncertainties (total - epistemic)
truth_dict = {}

# Load data for each event count
for n_event in n_events:
    if hardcuts:
        pwd = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/pt>1GeV/{n_event}/gnn/plots/uncalibrated/"
    else:
        pwd = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_event}/gnn/plots/uncalibrated/with_input_cut/"
    score_file = pwd + all_prefix + prefix[2] + f"{n_event}.txt"
    uncertainty_file = pwd + all_prefix + prefix[6] + f"{n_event}.txt"
    target_truth_file = pwd + all_prefix + prefix[4] + f"{n_event}.txt"
    non_target_truth_file = pwd + all_prefix + prefix[5] + f"{n_event}.txt"
    total_uncertainty_file = pwd + "Total_uncertainty_" + f"{n_event}.txt"
    epistemic_uncertainty_file = pwd + all_prefix + "epistemic_uncertainty_" + f"{n_event}.txt"
    
    # Load Standard deviation data
    scores_dict[n_event] = np.loadtxt(score_file)
    uncertainties_dict[n_event] = np.loadtxt(uncertainty_file)
    # print(np.loadtxt(target_truth_file))
    # print(np.loadtxt(non_target_truth_file))
    truth_file = np.loadtxt(target_truth_file, dtype=int) | np.loadtxt(non_target_truth_file, dtype=int)
    truth_dict[n_event] = truth_file
    
    # Load additional uncertainty types
    total_uncertainties_dict[n_event] = np.loadtxt(total_uncertainty_file)
    epistemic_uncertainties_dict[n_event] = np.loadtxt(epistemic_uncertainty_file)
    
    # Calculate aleatoric uncertainty (total - epistemic)
    aleatoric_uncertainties_dict[n_event] = total_uncertainties_dict[n_event] - epistemic_uncertainties_dict[n_event]
    
    print(f"Loaded data for {n_event} events")

# Combine all uncertainty dictionaries
uncertainties_dicts = {
    "Standard deviation": uncertainties_dict,
    "Total": total_uncertainties_dict,
    "Epistemic": epistemic_uncertainties_dict,
    "Aleatoric": aleatoric_uncertainties_dict
}

# Configuration for the plot
config = {
    "nb_MCD_passes": 100,
    "dropout": 0.1,
    "score_cut": 0.5  # Adjust as needed
}

if hardcuts:
    out_dir = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/pt>1GeV/uncertainty_comparison/uncalibrated/"
else:
    out_dir = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/uncertainty_comparison/uncalibrated/"
os.makedirs(out_dir, exist_ok=True)

# Generate the plot for all uncertainty types
uncertainty_types = ["Standard deviation", "Total", "Epistemic", "Aleatoric"]
plot_uncertainty_comparison(scores_dict, uncertainties_dicts, uncertainty_types, truth_dict, config, out_dir)

# Generate the overall uncertainty plots (without score binning)
plot_overall_uncertainty_comparison(scores_dict, uncertainties_dicts, uncertainty_types, truth_dict, config, out_dir)