import matplotlib.pyplot as plt
import numpy as np
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify

# Import all the uncertainty, scores and truth mask files
n_train = 1400
# Intrinsec GNN uncertainty

print("Loading data for intrinsic GNN uncertainty...")

intrinsic_gnn_bce_entropy = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_BCE_score_entropy_{n_train}.txt")

intrinsec_gnn_std_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_uncertainties_{n_train}.txt")
intrinsec_gnn_std_uncertainty_track = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_uncertainties_track_{n_train}.txt")
intrinsec_scores = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_scores_{n_train}.txt")
intrinsec_truth_target_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_target_truth_{n_train}.txt")
intrinsec_truth_non_target_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_non_target_truth_{n_train}.txt")
intrinsec_truth_target_track_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_target_truth_track_{n_train}.txt")
intrinsec_truth_non_target_track_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_non_target_truth_track_{n_train}.txt")
intrinsec_false_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_false_{n_train}.txt")
intrinsic_epistemic_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_epistemic_uncertainty_1400.txt")
intrinsic_total_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/Total_uncertainty_1400.txt")


# Load eta and pt data for intrinsic
intrinsec_eta = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_eta_{n_train}.txt")
intrinsec_pt = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/plots/uncalibrated/with_input_cut/all_flat_pt_{n_train}.txt")

# Convert truth masks to boolean arrays after loading
intrinsec_truth_target_mask = intrinsec_truth_target_mask.astype(bool)
intrinsec_truth_non_target_mask = intrinsec_truth_non_target_mask.astype(bool)
intrinsec_truth_target_track_mask = intrinsec_truth_target_track_mask.astype(bool)
intrinsec_truth_non_target_track_mask = intrinsec_truth_non_target_track_mask.astype(bool)
intrinsec_false_mask = intrinsec_false_mask.astype(bool)

print("Loading data for propagation GNN uncertainty...")


# propagation GNN uncertainty
prop_scores = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_scores_{n_train}.txt")
prop_truth_target_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_target_truth_{n_train}.txt")
prop_truth_non_target_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_non_target_truth_{n_train}.txt")
prop_truth_target_track_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_target_truth_track_{n_train}.txt")
prop_truth_non_target_track_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_non_target_truth_track_{n_train}.txt")
prop_false_mask = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_false_{n_train}.txt")
prop_std_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_uncertainties_{n_train}.txt")
prop_std_uncertainty_track = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_uncertainties_track_{n_train}.txt")
# Load eta and pt data for propagation
prop_eta = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_eta_{n_train}.txt")
prop_pt = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_pt_{n_train}.txt")
prop_epistemic_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/all_flat_epistemic_uncertainty_{n_train}.txt")
prop_total_uncertainty = np.loadtxt(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/GNN/Total_uncertainty_{n_train}.txt")

# Convert truth masks to boolean arrays after loading
prop_truth_target_mask = prop_truth_target_mask.astype(bool)
prop_truth_non_target_mask = prop_truth_non_target_mask.astype(bool)
prop_truth_target_track_mask = prop_truth_target_track_mask.astype(bool)
prop_truth_non_target_track_mask = prop_truth_non_target_track_mask.astype(bool)
prop_false_mask = prop_false_mask.astype(bool)

def plot_combined_uncertainty_vs_eta(intrinsic_eta, intrinsic_uncertainties, intrinsic_target_truth, intrinsic_non_target_truth, intrinsic_false,
                                    prop_eta, prop_uncertainties, prop_target_truth, prop_non_target_truth, prop_false,
                                    config, plot_config):
    """
    Plot the combined distribution of uncertainties vs. eta with error bands.
    Combines intrinsic and propagation uncertainties after binning them separately.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create eta bins
    eta_bins = np.linspace(-4, 4, 101)
    
    # Function to calculate binned statistics
    def calculate_binned_stats(eta_values, uncertainties, truth_mask):
        bin_indices = np.digitize(eta_values, eta_bins) - 1
        means = []
        errors = []
        bin_centers = []
        
        for i in range(len(eta_bins) - 1):
            mask = (bin_indices == i) & truth_mask
            if np.sum(mask) > 0:
                bin_uncertainties = uncertainties[mask]
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
                bin_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
            else:
                means.append(np.nan)
                errors.append(np.nan)
                bin_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(means), np.array(errors)
    
    # Calculate binned statistics for each edge type and dataset
    edge_types = [
        ("Target", intrinsic_target_truth, prop_target_truth, 'tab:blue'),
        ("Non-target", intrinsic_non_target_truth, prop_non_target_truth, 'tab:green'), 
        ("False", intrinsic_false, prop_false, 'tab:orange')
    ]
    
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        # Calculate binned statistics for intrinsic data
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_eta, intrinsic_uncertainties, intrinsic_mask
        )
        
        # Calculate binned statistics for propagation data
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_eta, prop_uncertainties, prop_mask
        )
        
        # Combine the binned results (sum means and propagate errors)
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(combined_means) & ~np.isnan(combined_errors)
        if np.any(valid_mask):
            ax.plot(intrinsic_bin_centers[valid_mask], combined_means[valid_mask], 
                   '-', linewidth=2, label=f'{edge_type} Edges', color=color)
            ax.fill_between(
                intrinsic_bin_centers[valid_mask], 
                combined_means[valid_mask] - combined_errors[valid_mask], 
                combined_means[valid_mask] + combined_errors[valid_mask], 
                alpha=0.3, color=color,
                edgecolor=None
            )
    
    # Set labels and formatting
    ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Combined Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.4)
    
    # Set y-axis limits based on data
    all_combined_data = []
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_eta, intrinsic_uncertainties, intrinsic_mask
        )
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_eta, prop_uncertainties, prop_mask
        )
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        all_combined_data.extend(combined_means[~np.isnan(combined_means)])
        all_combined_data.extend((combined_means + combined_errors)[~np.isnan(combined_means + combined_errors)])
    
    # if len(all_combined_data) > 0:
    #     ax.set_ylim(0, np.max(all_combined_data) * 1.1)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Apply ATLAS styling
    n_train = config.get("n_train", 1400)
    pt_min = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0] / 1e3
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + "Combined Intrinsic + Propagation Uncertainties" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes" + "\n"
        + f"Dropout rate: {config.get('hidden_dropout', 0.0)}" + "\n"
    )
    
    fig.tight_layout()
    
    # Save the figure
    save_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_eta_{n_train}.png"
    save_path_svg = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_eta_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Combined uncertainty vs. eta plot saved to {save_path}")


def plot_combined_uncertainty_vs_pt(intrinsic_pt, intrinsic_uncertainties, intrinsic_target_truth, intrinsic_non_target_truth, prop_pt, prop_uncertainties, prop_target_truth, prop_non_target_truth, config, plot_config):
    """
    Plot the combined distribution of uncertainties vs. pT for track edges with error bands.
    Combines intrinsic and propagation uncertainties after binning them separately.
    Note: Only plots target and non-target edges as pT is only available for track edges.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define PT bins based on units (following eval_utils pattern)
    pt_min = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0]
    pt_max = 50*1000
    if pt_min == 0:
        pt_min += 1e-1
    
    # Create bins for PT (log scale)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 101)
    
    # Function to calculate binned statistics
    def calculate_binned_stats(pt_values, uncertainties, truth_mask):
        bin_indices = np.digitize(pt_values, pt_bins) - 1
        means = []
        errors = []
        bin_centers = []
        
        for i in range(len(pt_bins) - 1):
            mask = (bin_indices == i) & truth_mask
            if np.sum(mask) > 0:
                bin_uncertainties = uncertainties[mask]
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
                bin_centers.append((pt_bins[i] + pt_bins[i+1]) / 2)
            else:
                means.append(np.nan)
                errors.append(np.nan)
                bin_centers.append((pt_bins[i] + pt_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(means), np.array(errors)
    
    # Calculate binned statistics for each edge type and dataset (only target and non-target for pT)
    edge_types = [
        ("Target", intrinsic_target_truth, prop_target_truth, 'tab:blue'),
        ("Non-target", intrinsic_non_target_truth, prop_non_target_truth, 'tab:green')
    ]
    
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        # Calculate binned statistics for intrinsic data
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_pt, intrinsic_uncertainties, intrinsic_mask
        )
        
        # Calculate binned statistics for propagation data
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_pt, prop_uncertainties, prop_mask
        )
        
        # Combine the binned results (sum means and propagate errors)
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(combined_means) & ~np.isnan(combined_errors)
        if np.any(valid_mask):
            ax.plot(intrinsic_bin_centers[valid_mask], combined_means[valid_mask], 
                   '-', linewidth=2, label=f'{edge_type} Track Edges', color=color)
            ax.fill_between(
                intrinsic_bin_centers[valid_mask], 
                combined_means[valid_mask] - combined_errors[valid_mask], 
                combined_means[valid_mask] + combined_errors[valid_mask], 
                alpha=0.3, color=color,
                edgecolor=None
            )
    
    # Set labels and formatting
    ax.set_xscale('log')
    ax.set_xlabel('$p_T$ [MeV]', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Combined Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(pt_min, pt_max)
    ax.set_ylim(0, 0.4)
    
    # Set y-axis limits based on data
    all_combined_data = []
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_pt, intrinsic_uncertainties, intrinsic_mask
        )
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_pt, prop_uncertainties, prop_mask
        )
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        all_combined_data.extend(combined_means[~np.isnan(combined_means)])
        all_combined_data.extend((combined_means + combined_errors)[~np.isnan(combined_means + combined_errors)])
    
    # if len(all_combined_data) > 0:
    #     ax.set_ylim(0, np.max(all_combined_data) * 1.1)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Apply ATLAS styling
    n_train = config.get("n_train", 1400)
    pt_min_legend = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0] / 1e3
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min_legend}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + "Combined Intrinsic + Propagation Uncertainties" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes" + "\n"
        + f"Dropout rate: {config.get('hidden_dropout', 0.0)}" + "\n"
    )
    
    fig.tight_layout()
    
    # Save the figure
    save_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_pt_{n_train}.png"
    save_path_svg = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_pt_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Combined uncertainty vs. pT plot saved to {save_path}")


def plot_combined_uncertainty_vs_score(intrinsic_scores, intrinsic_uncertainties, intrinsic_target_truth, intrinsic_non_target_truth, intrinsic_false,
                                     prop_scores, prop_uncertainties, prop_target_truth, prop_non_target_truth, prop_false,
                                     config, plot_config):
    """
    Plot the combined distribution of uncertainties vs. edge scores with error bands.
    Combines intrinsic and propagation uncertainties after binning them separately.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)
    
    # Function to calculate binned statistics
    def calculate_binned_stats(scores, uncertainties, truth_mask):
        bin_indices = np.digitize(scores, score_bins) - 1
        means = []
        errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            mask = (bin_indices == i) & truth_mask
            if np.sum(mask) > 0:
                bin_uncertainties = uncertainties[mask]
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
            else:
                means.append(np.nan)
                errors.append(np.nan)
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(means), np.array(errors)
    
    # Calculate binned statistics for each edge type and dataset
    edge_types = [
        ("Target", intrinsic_target_truth, prop_target_truth, 'tab:blue'),
        ("Non-target", intrinsic_non_target_truth, prop_non_target_truth, 'tab:green'), 
        ("False", intrinsic_false, prop_false, 'tab:orange')
    ]
    
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        # Calculate binned statistics for intrinsic data
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_scores, intrinsic_uncertainties, intrinsic_mask
        )
        
        # Calculate binned statistics for propagation data
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_scores, prop_uncertainties, prop_mask
        )
        
        # Combine the binned results (sum means and propagate errors)
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(combined_means) & ~np.isnan(combined_errors)
        if np.any(valid_mask):
            ax.plot(intrinsic_bin_centers[valid_mask], combined_means[valid_mask], 
                   '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax.fill_between(
                intrinsic_bin_centers[valid_mask], 
                combined_means[valid_mask] - combined_errors[valid_mask], 
                combined_means[valid_mask] + combined_errors[valid_mask], 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
    
    # Set labels and formatting
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Combined Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    
    # Set y-axis limits based on data
    all_combined_data = []
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_scores, intrinsic_uncertainties, intrinsic_mask
        )
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_scores, prop_uncertainties, prop_mask
        )
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        all_combined_data.extend(combined_means[~np.isnan(combined_means)])
        all_combined_data.extend((combined_means + combined_errors)[~np.isnan(combined_means + combined_errors)])
    
    # if len(all_combined_data) > 0:
    #     ax.set_ylim(0, np.max(all_combined_data) * 1.1)
    
    # Add vertical line at score cut if available
    if "score_cut" in config:
        score_cut = config["score_cut"]
        ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Apply ATLAS styling
    n_train = config.get("n_train", 1400)
    pt_min = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0] / 1e3
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + "Combined Intrinsic + Propagation Uncertainties" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes" + "\n"
        + f"Dropout rate: {config.get('hidden_dropout', 0.0)}" + "\n"
    )
    
    fig.tight_layout()
    
    # Save the figure
    calib_folder = "calibrated" if config.get("calibration", False) else "uncalibrated"
    if not config.get("input_cut", True):
        calib_folder += "/no_input_cut"
    else:
        calib_folder += "/with_input_cut"
    
    save_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_score_{n_train}.png"
    save_path_svg = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/combined_uncertainty_vs_score_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Combined uncertainty vs. score plot saved to {save_path}")


def plot_combined_epistemic_uncertainty_vs_score(intrinsic_scores, intrinsic_epistemic_uncertainty, intrinsic_target_truth, intrinsic_non_target_truth, intrinsic_false,
                                     prop_scores, prop_epistemic_uncertainty, prop_target_truth, prop_non_target_truth, prop_false,
                                     config, plot_config):
    """
    Plot the combined distribution of epistemic uncertainties vs. edge scores with error bands.
    Combines intrinsic and propagation epistemic uncertainties after binning them separately.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)
    
    # Function to calculate binned statistics
    def calculate_binned_stats(scores, uncertainties, truth_mask):
        bin_indices = np.digitize(scores, score_bins) - 1
        means = []
        errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            mask = (bin_indices == i) & truth_mask
            if np.sum(mask) > 0:
                bin_uncertainties = uncertainties[mask]
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
            else:
                means.append(np.nan)
                errors.append(np.nan)
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(means), np.array(errors)
    
    # Calculate binned statistics for each edge type and dataset
    edge_types = [
        ("Target", intrinsic_target_truth, prop_target_truth, 'tab:blue'),
        ("Non-target", intrinsic_non_target_truth, prop_non_target_truth, 'tab:green'), 
        ("False", intrinsic_false, prop_false, 'tab:orange')
    ]
    
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        # Calculate binned statistics for intrinsic data
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_scores, intrinsic_epistemic_uncertainty, intrinsic_mask
        )
        
        # Calculate binned statistics for propagation data
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_scores, prop_epistemic_uncertainty, prop_mask
        )
        
        # Combine the binned results (sum means and propagate errors)
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(combined_means) & ~np.isnan(combined_errors)
        if np.any(valid_mask):
            ax.plot(intrinsic_bin_centers[valid_mask], combined_means[valid_mask], 
                   '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax.fill_between(
                intrinsic_bin_centers[valid_mask], 
                combined_means[valid_mask] - combined_errors[valid_mask], 
                combined_means[valid_mask] + combined_errors[valid_mask], 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
    
    # Set labels and formatting
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Combined Epistemic Uncertainty', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.6)
    
    # Add vertical line at score cut if available
    if "score_cut" in config:
        score_cut = config["score_cut"]
        ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Apply ATLAS styling
    n_train = config.get("n_train", 1400)
    pt_min = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0] / 1e3
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + "Combined Intrinsic + Propagation Epistemic Uncertainties" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes" + "\n"
        + f"Dropout rate: {config.get('hidden_dropout', 0.0)}" + "\n"
    )
    
    fig.tight_layout()
    
    # Save the figure
    save_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/uncertainty_combination/combined_epistemic_uncertainty_vs_score_{n_train}.png"
    save_path_svg = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/uncertainty_combination/combined_epistemic_uncertainty_vs_score_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Combined epistemic uncertainty vs. score plot saved to {save_path}")

def plot_combined_total_uncertainty_vs_score(intrinsic_scores, intrinsic_total_uncertainty, intrinsic_target_truth, intrinsic_non_target_truth, intrinsic_false,
                                     prop_scores, prop_total_uncertainty, prop_target_truth, prop_non_target_truth, prop_false,
                                     config, plot_config):
    """
    Plot the combined distribution of total uncertainties vs. edge scores with error bands.
    Combines intrinsic and propagation total uncertainties after binning them separately.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)
    
    # Function to calculate binned statistics
    def calculate_binned_stats(scores, uncertainties, truth_mask):
        bin_indices = np.digitize(scores, score_bins) - 1
        means = []
        errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            mask = (bin_indices == i) & truth_mask
            if np.sum(mask) > 0:
                bin_uncertainties = uncertainties[mask]
                means.append(np.mean(bin_uncertainties))
                errors.append(np.std(bin_uncertainties) / np.sqrt(len(bin_uncertainties)))
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
            else:
                means.append(np.nan)
                errors.append(np.nan)
                bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(means), np.array(errors)
    
    # Calculate binned statistics for each edge type and dataset
    edge_types = [
        ("Target", intrinsic_target_truth, prop_target_truth, 'tab:blue'),
        ("Non-target", intrinsic_non_target_truth, prop_non_target_truth, 'tab:green'), 
        ("False", intrinsic_false, prop_false, 'tab:orange')
    ]
    
    for edge_type, intrinsic_mask, prop_mask, color in edge_types:
        # Calculate binned statistics for intrinsic data
        intrinsic_bin_centers, intrinsic_means, intrinsic_errors = calculate_binned_stats(
            intrinsic_scores, intrinsic_total_uncertainty, intrinsic_mask
        )
        
        # Calculate binned statistics for propagation data
        prop_bin_centers, prop_means, prop_errors = calculate_binned_stats(
            prop_scores, prop_total_uncertainty, prop_mask
        )
        
        # Combine the binned results (sum means and propagate errors)
        combined_means = np.nansum([intrinsic_means, prop_means], axis=0)
        combined_errors = np.sqrt(np.nansum([intrinsic_errors**2, prop_errors**2], axis=0))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(combined_means) & ~np.isnan(combined_errors)
        if np.any(valid_mask):
            ax.plot(intrinsic_bin_centers[valid_mask], combined_means[valid_mask], 
                   '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax.fill_between(
                intrinsic_bin_centers[valid_mask], 
                combined_means[valid_mask] - combined_errors[valid_mask], 
                combined_means[valid_mask] + combined_errors[valid_mask], 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
    
    # Set labels and formatting
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Combined Total Uncertainty', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.6)
    
    # Add vertical line at score cut if available
    if "score_cut" in config:
        score_cut = config["score_cut"]
        ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Apply ATLAS styling
    n_train = config.get("n_train", 1400)
    pt_min = config.get("target_tracks", {}).get('track_particle_pt', [1000])[0] / 1e3
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + "Combined Intrinsic + Propagation Total Uncertainties" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes" + "\n"
        + f"Dropout rate: {config.get('hidden_dropout', 0.0)}" + "\n"
    )
    
    fig.tight_layout()
    
    # Save the figure
    save_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/uncertainty_combination/combined_total_uncertainty_vs_score_{n_train}.png"
    save_path_svg = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/uncertainty_combination/combined_total_uncertainty_vs_score_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Combined total uncertainty vs. score plot saved to {save_path}")

# Add these function calls after loading all the data
# print("Creating combined uncertainty vs score plot...")
# plot_combined_uncertainty_vs_score(
#     intrinsec_scores, intrinsec_gnn_std_uncertainty, intrinsec_truth_target_mask, intrinsec_truth_non_target_mask, intrinsec_false_mask,
#     prop_scores, prop_std_uncertainty, prop_truth_target_mask, prop_truth_non_target_mask, prop_false_mask,
#     {"n_train": n_train, "score_cut": 0.5, "target_tracks": {"track_particle_pt": [1000]}, "nb_MCD_passes": 100, "hidden_dropout": 0.1},
#     {}
# )

# Create the combined epistemic uncertainty vs score plots
# print("Creating combined epistemic uncertainty vs score plot...")
# plot_combined_epistemic_uncertainty_vs_score(
#     intrinsec_scores, intrinsic_epistemic_uncertainty, intrinsec_truth_target_mask, intrinsec_truth_non_target_mask, intrinsec_false_mask,
#     prop_scores, prop_epistemic_uncertainty, prop_truth_target_mask, prop_truth_non_target_mask, prop_false_mask,
#     {"n_train": n_train, "score_cut": 0.5, "target_tracks": {"track_particle_pt": [1000]}, "nb_MCD_passes": 100, "hidden_dropout": 0.1},
#     {}
# )

# Create the combined total uncertainty vs score plots
# print("Creating combined total uncertainty vs score plot...")
# plot_combined_total_uncertainty_vs_score(
#     intrinsec_scores, intrinsic_total_uncertainty, intrinsec_truth_target_mask, intrinsec_truth_non_target_mask, intrinsec_false_mask,
#     prop_scores, prop_total_uncertainty, prop_truth_target_mask, prop_truth_non_target_mask, prop_false_mask,
#     {"n_train": n_train, "score_cut": 0.5, "target_tracks": {"track_particle_pt": [1000]}, "nb_MCD_passes": 100, "hidden_dropout": 0.1},
#     {}
# )

# Create the combined uncertainty vs eta and pT plots

# print("Creating combined uncertainty vs eta plot...")
# plot_combined_uncertainty_vs_eta(
#     intrinsec_eta, intrinsec_gnn_std_uncertainty, intrinsec_truth_target_mask, intrinsec_truth_non_target_mask, intrinsec_false_mask,
#     prop_eta, prop_std_uncertainty, prop_truth_target_mask, prop_truth_non_target_mask, prop_false_mask,
#     {"n_train": n_train, "target_tracks": {"track_particle_pt": [1000]}, "nb_MCD_passes": 100, "hidden_dropout": 0.1},
#     {}
# )

print("Creating combined uncertainty vs pT plot...")
plot_combined_uncertainty_vs_pt(
    intrinsec_pt, intrinsec_gnn_std_uncertainty_track, intrinsec_truth_target_track_mask, intrinsec_truth_non_target_track_mask,
    prop_pt, prop_std_uncertainty_track, prop_truth_target_track_mask, prop_truth_non_target_track_mask,
    {"n_train": n_train, "target_tracks": {"track_particle_pt": [1000]}, "nb_MCD_passes": 100, "hidden_dropout": 0.1},
    {}
)

