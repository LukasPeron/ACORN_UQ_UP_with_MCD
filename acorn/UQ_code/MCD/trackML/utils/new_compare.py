import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import os
from matplotlib.lines import Line2D

def plot_overall_uncertainty_comparison(scores_dict, uncertainties_dicts, prop_uncertainties_dicts, uncertainty_types, truth_dict, prop_truth_dict, config, out_dir):
    """
    Plot mean uncertainties across different train event counts WITHOUT score binning.
    Creates 9 plots: for each truth class (target, non-target, false) it outputs 3 plots:
    intrinsic uncertainties, propagation uncertainties, combined uncertainties.
    
    Parameters:
    - scores_dict: Dictionary mapping n_event to score arrays
    - uncertainties_dicts: Dictionary mapping uncertainty type to dictionaries of n_event to uncertainty arrays (intrinsic)
    - prop_uncertainties_dicts: Dictionary mapping uncertainty type to dictionaries of n_event to uncertainty arrays (propagation)
    - uncertainty_types: List of uncertainty types to plot
    - truth_dict: Dictionary mapping n_event to truth arrays (intrinsic)
    - prop_truth_dict: Dictionary mapping n_event to truth arrays (propagation)
    - config: Configuration dictionary
    - out_dir: Output directory for saving plots
    """
    # Sort event counts for x-axis
    sorted_events = sorted(scores_dict.keys())
    
    # Define truth classes
    truth_classes = [
        ("target", "Target Truth"),
        ("non_target", "Non-target Truth"),
        ("false", "False")
    ]
    
    # Define uncertainty source groups
    uncertainty_sources = [
        ("intrinsic", "Intrinsic", uncertainties_dicts, truth_dict),
        ("propagation", "Propagation", prop_uncertainties_dicts, prop_truth_dict),
        ("combined", "Combined", {
            "Standard deviation": uncertainties_dicts.get("Combined Standard deviation", {}),
            "Total": uncertainties_dicts.get("Combined Total", {}),
            "Epistemic": uncertainties_dicts.get("Combined Epistemic", {}),
            "Aleatoric": uncertainties_dicts.get("Combined Aleatoric", {})
        }, truth_dict)  # Use intrinsic truth for combined plots
    ]
    
    # Create plots for each truth class and uncertainty source combination
    for truth_class, truth_label in truth_classes:
        for source_key, source_label, source_uncertainties_dicts, source_truth_dict in uncertainty_sources:
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # For each uncertainty type, plot a line
            for uncertainty_type in uncertainty_types:
                # Skip combined types for intrinsic and propagation plots
                if source_key in ["intrinsic", "propagation"] and "Combined" in uncertainty_type:
                    continue
                
                # For combined plots, use the base uncertainty type
                plot_uncertainty_type = uncertainty_type
                if source_key == "combined":
                    if uncertainty_type not in source_uncertainties_dicts or not source_uncertainties_dicts[uncertainty_type]:
                        continue
                
                # Skip if uncertainty type doesn't exist in the dictionary
                if plot_uncertainty_type not in source_uncertainties_dicts:
                    print(f"Warning: {plot_uncertainty_type} not found in {source_key} uncertainties_dicts")
                    continue
                    
                uncertainties_dict = source_uncertainties_dicts[plot_uncertainty_type]
                
                # Arrays to hold mean uncertainties for each event count
                means = []
                errors = []
                
                # Calculate mean uncertainties for each event count
                for n_event in sorted_events:
                    # Skip if this n_event doesn't exist for this uncertainty type
                    if n_event not in uncertainties_dict or n_event not in source_truth_dict:
                        means.append(np.nan)
                        errors.append(np.nan)
                        continue
                        
                    uncertainties = uncertainties_dict[n_event]
                    truth_masks = source_truth_dict[n_event]
                    
                    # Apply truth class mask based on the specific truth class
                    if truth_class == "target":
                        mask = truth_masks["target"]
                    elif truth_class == "non_target":
                        mask = truth_masks["non_target"]
                    else:  # false
                        mask = truth_masks["false"]
                        
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
                
                # Define colors for different uncertainty types
                colors = {
                    "Standard deviation": "blue", 
                    "Total": "red", 
                    "Epistemic": "green", 
                    "Aleatoric": "purple"
                }
                
                # Use default color if not in dictionary
                color = colors.get(uncertainty_type, "black")
                
                # Plot with error bands
                ax.plot(sorted_events, means, '-o', color=color, linewidth=2, 
                        label=f'{uncertainty_type}')
                ax.fill_between(
                    sorted_events, 
                    means - errors, 
                    means + errors, 
                    alpha=0.3,
                    color=color,
                    edgecolor=None
                )
            
            # Configure axis
            ax.set_xlabel('Number of Training Events', fontsize=14, ha="right", x=0.95)
            ax.set_ylabel('Uncertainty', fontsize=14, ha="right", y=0.95)
            ax.set_ylim(0, 0.4)  # Adjust y-axis limits as needed
            # Set x-axis ticks to be exactly at our event counts
            ax.set_xticks(sorted_events)
            ax.set_xticklabels(sorted_events)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=12)
            
            # Apply ATLAS styling
            atlasify(" ",
                rf"$p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {config.get('dropout', 0.0)}"
                + "\n"
                + f"Evaluated on 50 events in valset" + "\n"
                f"{truth_label} edges - {source_label} uncertainties"
            )
            
            fig.tight_layout()
            
            # Save the figure
            save_path = os.path.join(out_dir, f"{source_key}_uncertainties_vs_events_{truth_class}_edges.png")
            save_path_svg = os.path.join(out_dir, f"{source_key}_uncertainties_vs_events_{truth_class}_edges.svg")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            fig.savefig(save_path_svg, bbox_inches='tight')
            plt.close(fig)
            print(f"{source_label} uncertainties for {truth_label} edges plot saved to {save_path}")

hardcuts = False  # Set to True for hard cuts
if hardcuts:
    n_events = [0, 50, 100, 200, 400, 800, 1400]
else:
    n_events = [100, 200, 400, 800, 1400]
prefix = ["scores_", "target_truth_", "non_target_truth_", "uncertainties_"]
all_prefix = "all_flat_"

# Prepare the data dictionaries for intrinsic uncertainties
scores_dict = {}
uncertainties_dict = {}  # Standard deviation uncertainties
total_uncertainties_dict = {}  # Total uncertainties
epistemic_uncertainties_dict = {}  # Epistemic uncertainties
aleatoric_uncertainties_dict = {}  # Aleatoric uncertainties (total - epistemic)
truth_dict = {}

# Prepare dictionaries for propagation uncertainties
prop_uncertainties_dict = {}
prop_total_uncertainties_dict = {}
prop_epistemic_uncertainties_dict = {}
prop_aleatoric_uncertainties_dict = {}
prop_truth_dict = {}

# Load intrinsic data for each event count
for n_event in n_events:
    if hardcuts:
        pwd = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/pt>1GeV/{n_event}/gnn/plots/uncalibrated/"
    else:
        pwd = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_event}/gnn/plots/uncalibrated/with_input_cut/"
    score_file = pwd + all_prefix + prefix[0] + f"{n_event}.txt"
    uncertainty_file = pwd + all_prefix + prefix[3] + f"{n_event}.txt"
    target_truth_file = pwd + all_prefix + prefix[1] + f"{n_event}.txt"
    non_target_truth_file = pwd + all_prefix + prefix[2] + f"{n_event}.txt"
    false_file = pwd + all_prefix + "false_" + f"{n_event}.txt"
    total_uncertainty_file = pwd + "Total_uncertainty_" + f"{n_event}.txt"
    epistemic_uncertainty_file = pwd + all_prefix + "epistemic_uncertainty_" + f"{n_event}.txt"
    
    # Load Standard deviation data
    scores_dict[n_event] = np.loadtxt(score_file)
    uncertainties_dict[n_event] = np.loadtxt(uncertainty_file)
    
    # Load truth masks explicitly (fix the deprecation warning)
    target_truth = np.loadtxt(target_truth_file).astype(bool)
    non_target_truth = np.loadtxt(non_target_truth_file).astype(bool)
    false_truth = np.loadtxt(false_file).astype(bool)
    
    truth_dict[n_event] = {
        "target": target_truth,
        "non_target": non_target_truth,
        "false": false_truth
    }
    
    # Load additional uncertainty types
    total_uncertainties_dict[n_event] = np.loadtxt(total_uncertainty_file)
    epistemic_uncertainties_dict[n_event] = np.loadtxt(epistemic_uncertainty_file)
    
    # Calculate aleatoric uncertainty (total - epistemic)
    aleatoric_uncertainties_dict[n_event] = total_uncertainties_dict[n_event] - epistemic_uncertainties_dict[n_event]
    
    print(f"Loaded intrinsic data for {n_event} events")

# Load propagation data for all n_events (not just 1400)
for n_event in n_events:
    prop_pwd = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_event}/UQ_propagation/"
    try:
        prop_uncertainty_file = prop_pwd + all_prefix + prefix[3] + f"{n_event}.txt"
        prop_total_uncertainty_file = prop_pwd + "Total_uncertainty_" + f"{n_event}.txt"  
        prop_epistemic_uncertainty_file = prop_pwd + all_prefix + "epistemic_uncertainty_" + f"{n_event}.txt"
        prop_target_truth_file = prop_pwd + all_prefix + prefix[1] + f"{n_event}.txt"
        prop_non_target_truth_file = prop_pwd + all_prefix + prefix[2] + f"{n_event}.txt"
        prop_false_file = prop_pwd + all_prefix + "false_" + f"{n_event}.txt"
        
        prop_uncertainties_dict[n_event] = np.loadtxt(prop_uncertainty_file)
        prop_total_uncertainties_dict[n_event] = np.loadtxt(prop_total_uncertainty_file)
        prop_epistemic_uncertainties_dict[n_event] = np.loadtxt(prop_epistemic_uncertainty_file)
        prop_aleatoric_uncertainties_dict[n_event] = prop_total_uncertainties_dict[n_event] - prop_epistemic_uncertainties_dict[n_event]
        
        # Load propagation truth masks explicitly
        prop_target_truth = np.loadtxt(prop_target_truth_file).astype(bool)
        prop_non_target_truth = np.loadtxt(prop_non_target_truth_file).astype(bool)
        prop_false_truth = np.loadtxt(prop_false_file).astype(bool)
        
        prop_truth_dict[n_event] = {
            "target": prop_target_truth,
            "non_target": prop_non_target_truth,
            "false": prop_false_truth
        }
        
        print(f"Loaded propagation data for {n_event} events")
    except FileNotFoundError as e:
        print(f"Warning: Could not load propagation data for {n_event} events - {e}")
        # Don't create empty arrays if files don't exist
        continue

# Combine all uncertainty dictionaries
uncertainties_dicts = {
    "Standard deviation": uncertainties_dict,
    "Total": total_uncertainties_dict,
    "Epistemic": epistemic_uncertainties_dict,
    "Aleatoric": aleatoric_uncertainties_dict
}

# Combine propagation uncertainty dictionaries
prop_uncertainties_dicts = {
    "Standard deviation": prop_uncertainties_dict,
    "Total": prop_total_uncertainties_dict,
    "Epistemic": prop_epistemic_uncertainties_dict,
    "Aleatoric": prop_aleatoric_uncertainties_dict
}

# Add combined uncertainties (sum of intrinsic and propagation averages)
combined_uncertainties_dict = {}
combined_total_uncertainties_dict = {}
combined_epistemic_uncertainties_dict = {}
combined_aleatoric_uncertainties_dict = {}

for n_event in n_events:
    if n_event in prop_uncertainties_dict:  # Only combine if we have propagation data
        # For combined uncertainties, we need to average each type first, then sum
        # This creates arrays where each element is the sum of the respective averages
        intrinsic_std = uncertainties_dict[n_event]
        prop_std = prop_uncertainties_dict[n_event]
        
        # Create combined arrays by adding the averages
        combined_uncertainties_dict[n_event] = intrinsic_std + np.full_like(intrinsic_std, np.mean(prop_std))
        combined_total_uncertainties_dict[n_event] = total_uncertainties_dict[n_event] + np.full_like(total_uncertainties_dict[n_event], np.mean(prop_total_uncertainties_dict[n_event]))
        combined_epistemic_uncertainties_dict[n_event] = epistemic_uncertainties_dict[n_event] + np.full_like(epistemic_uncertainties_dict[n_event], np.mean(prop_epistemic_uncertainties_dict[n_event]))
        combined_aleatoric_uncertainties_dict[n_event] = aleatoric_uncertainties_dict[n_event] + np.full_like(aleatoric_uncertainties_dict[n_event], np.mean(prop_aleatoric_uncertainties_dict[n_event]))
    else:
        # For events without propagation data, just use intrinsic uncertainties
        combined_uncertainties_dict[n_event] = uncertainties_dict[n_event]
        combined_total_uncertainties_dict[n_event] = total_uncertainties_dict[n_event]
        combined_epistemic_uncertainties_dict[n_event] = epistemic_uncertainties_dict[n_event]
        combined_aleatoric_uncertainties_dict[n_event] = aleatoric_uncertainties_dict[n_event]

# Add combined uncertainties to the dictionaries
uncertainties_dicts["Combined Standard deviation"] = combined_uncertainties_dict
uncertainties_dicts["Combined Total"] = combined_total_uncertainties_dict
uncertainties_dicts["Combined Epistemic"] = combined_epistemic_uncertainties_dict
uncertainties_dicts["Combined Aleatoric"] = combined_aleatoric_uncertainties_dict

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

# Generate the overall uncertainty plots (without score binning)
plot_overall_uncertainty_comparison(scores_dict, uncertainties_dicts, prop_uncertainties_dicts, 
                                  ["Standard deviation", "Total", "Epistemic", "Aleatoric"], 
                                  truth_dict, prop_truth_dict, config, out_dir)
