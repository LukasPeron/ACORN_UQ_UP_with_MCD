"""
The goal of this script is to implement the following algorithm:
1. load the input dataset for the filter (output of the metric learning)
2. do T MCD forward passes on it
3. for each edge of each graph, compute the mean predicted scores and the standard deviation of the predicted scores. Also remember their event and edge id
4. After all the MCD is done, we do one deterministic forward pass with the filter. We apply the input edge score cut to get rid of the lower score edges.
5. we do 100 MCD forward passes with the GNN and compute the mean and standard deviation of the predicted scores for each edge.
6. For the edges that are present in the deterministic filter output with a score above the cut, we plot the mean and standard deviation of the GNN scores against the mean and standard deviation of the filter scores and save the plots and the data in txt files (we stock the mean scores and the standard deviations in a good order (so that we can plot them easily later) for both filter and gnn).
"""

from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify

# Configuration
n_train = 1400
output_dir = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn_vs_filter/")
output_dir.mkdir(parents=True, exist_ok=True)

filter_dataset_name = "valset"  
gnn_dataset_name = "valset" 
n_mcd_passes = 100
filter_score_cut = 0.05  # Score cut for filter output

# Load filter model
filter_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/filter/artifacts/"
if list(Path(filter_ckpt_path).glob("best*"))!=[]:   
    filter_checkpoint = list(Path(filter_ckpt_path).glob("best*"))[0]
    filter_model = Filter.load_from_checkpoint(filter_checkpoint)
    filter_model.hparams["data_split"] = [0,50,0]
    filter_model.setup(stage="test", input_dir="input_dir")
    filter_dataset = getattr(filter_model, filter_dataset_name)
    print(f"Filter model loaded from {filter_checkpoint}")

# Load GNN model
gnn_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/artifacts/"
if list(Path(gnn_ckpt_path).glob("best*"))!=[]:
    gnn_checkpoint = list(Path(gnn_ckpt_path).glob("best*"))[0]
    gnn_model = InteractionGNN.load_from_checkpoint(gnn_checkpoint)
    gnn_model.hparams["data_split"] = [0,50,0]
    gnn_model.setup(stage="predict")
    gnn_model.hparams["input_cut"] = filter_score_cut
    gnn_dataset = getattr(gnn_model, gnn_dataset_name)
    print(f"GNN model loaded from {gnn_checkpoint}")

# Filter weights (from UQ_propagation.py)
filter_true_target_weight = 2
filter_true_non_target_weight = 0
filter_false_weight = 1

print("Step 1-3: Processing filter with MCD passes...")
# Step 1-3: MCD passes on filter
filter_edges_dict = {}

for t in tqdm(range(n_mcd_passes), desc="Filter MCD passes"):
    for num_event, event in enumerate(filter_dataset):
        with torch.inference_mode():
            filter_model.train()  # Enable dropout
            
            # Apply filter model stochastically
            eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
            event_filter = eval_dict_filter["batch"]
            
            # Get edge information
            edge_index = event_filter.edge_index.cpu().numpy()
            edge_scores = event_filter.edge_scores.cpu().numpy()
            edge_y = event_filter.edge_y.cpu().numpy()
            edge_weights = event_filter.edge_weights.cpu().numpy()
            
            # Store edge information
            for i in range(edge_index.shape[1]):
                edge_key = (num_event, edge_index[0, i].item(), edge_index[1, i].item())
                if edge_key not in filter_edges_dict:
                    filter_edges_dict[edge_key] = {
                        "scores": [], 
                        "y": edge_y[i], 
                        "weights": edge_weights[i], 
                    }
                filter_edges_dict[edge_key]["scores"].append(edge_scores[i])

print("Step 4: Deterministic filter pass with score cut...")
# Step 4: Deterministic filter pass to identify surviving edges
surviving_edges = set()
filter_model.eval()  # Disable dropout for deterministic pass

for num_event, event in enumerate(filter_dataset):
    with torch.inference_mode():
        eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
        event_filter = eval_dict_filter["batch"]
        
        edge_index = event_filter.edge_index.cpu().numpy()
        edge_scores = event_filter.edge_scores.cpu().numpy()
        
        # Apply score cut and identify surviving edges
        for i in range(edge_index.shape[1]):
            if edge_scores[i] >= filter_score_cut:
                edge_key = (num_event, edge_index[0, i].item(), edge_index[1, i].item())
                surviving_edges.add(edge_key)

print(f"Number of surviving edges after filter cut: {len(surviving_edges)}")

print("Step 5: Processing GNN with MCD passes on surviving edges...")
# Step 5: MCD passes on GNN for surviving edges
gnn_edges_dict = {}

for t in tqdm(range(n_mcd_passes), desc="GNN MCD passes"):
    for num_event, event in enumerate(filter_dataset):
        with torch.inference_mode():
            filter_model.eval()  # Deterministic filter
            gnn_model.train()    # Enable GNN dropout
            
            # Apply deterministic filter
            eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
            event_filter = eval_dict_filter["batch"]
            
            # Apply score cut
            # edge_mask = event_filter.edge_scores >= filter_score_cut
            # if edge_mask.sum() == 0:
            #     continue
                
            # Filter edges based on score cut
            # event_filter.edge_index = event_filter.edge_index[:, edge_mask]
            # event_filter.edge_scores = event_filter.edge_scores[edge_mask]
            # event_filter.edge_y = event_filter.edge_y[edge_mask]
            # event_filter.edge_weights = event_filter.edge_weights[edge_mask]
            
            # Process through GNN
            event_filter = gnn_dataset.handle_edge_list(event_filter.cpu())
            gnn_dict = gnn_model.shared_evaluation(event_filter.to(gnn_model.device), 0)
            event_gnn = gnn_dict["batch"]
            
            # Get GNN edge information
            edge_index = event_gnn.edge_index.cpu().numpy()
            edge_scores = event_gnn.edge_scores.cpu().numpy()
            edge_y = event_gnn.edge_y.cpu().numpy()
            edge_weights = event_gnn.edge_weights.cpu().numpy()
            
            # Store GNN edge information for surviving edges
            for i in range(edge_index.shape[1]):
                edge_key = (num_event, edge_index[0, i].item(), edge_index[1, i].item())
                if edge_key in surviving_edges:  # Only store if edge survived filter cut
                    if edge_key not in gnn_edges_dict:
                        gnn_edges_dict[edge_key] = {
                            "scores": [], 
                            "y": edge_y[i], 
                            "weights": edge_weights[i], 
                        }
                    gnn_edges_dict[edge_key]["scores"].append(edge_scores[i])

print("Computing statistics for matched edges...")
# Compute statistics for edges present in both filter and GNN outputs
matched_edges = set(filter_edges_dict.keys()) & set(gnn_edges_dict.keys()) & surviving_edges

print(f"Number of matched edges: {len(matched_edges)}")

# Prepare data for plotting
filter_mean_scores = []
filter_uncertainties = []
gnn_mean_scores = []
gnn_uncertainties = []
edge_labels = []  # target_truth, non_target_truth, false

for edge_key in matched_edges:
    # Filter statistics
    filter_scores = np.array(filter_edges_dict[edge_key]["scores"])
    filter_mean_scores.append(np.mean(filter_scores))
    filter_uncertainties.append(np.std(filter_scores))
    
    # GNN statistics
    gnn_scores = np.array(gnn_edges_dict[edge_key]["scores"])
    gnn_mean_scores.append(np.mean(gnn_scores))
    gnn_uncertainties.append(np.std(gnn_scores))
    
    # Edge classification
    edge_y = filter_edges_dict[edge_key]["y"]
    edge_weight = filter_edges_dict[edge_key]["weights"]
    
    if edge_y and edge_weight == filter_true_target_weight:
        edge_labels.append("target")
    elif edge_y and edge_weight == filter_true_non_target_weight:
        edge_labels.append("non_target")
    elif not edge_y and edge_weight == filter_false_weight:
        edge_labels.append("false")
    else:
        edge_labels.append("other")
    

# Convert to numpy arrays
filter_mean_scores = np.array(filter_mean_scores)
filter_uncertainties = np.array(filter_uncertainties)
gnn_mean_scores = np.array(gnn_mean_scores)
gnn_uncertainties = np.array(gnn_uncertainties)
edge_labels = np.array(edge_labels)

# Create masks for different edge types
target_mask = edge_labels == "target"
non_target_mask = edge_labels == "non_target"
false_mask = edge_labels == "false"

print("Step 6: Saving data and creating plots...")

# Save data to txt files
np.savetxt(output_dir / f"filter_mean_scores_{n_train}.txt", filter_mean_scores)
np.savetxt(output_dir / f"filter_uncertainties_{n_train}.txt", filter_uncertainties)
np.savetxt(output_dir / f"gnn_mean_scores_{n_train}.txt", gnn_mean_scores)
np.savetxt(output_dir / f"gnn_uncertainties_{n_train}.txt", gnn_uncertainties)
np.savetxt(output_dir / f"target_mask_{n_train}.txt", target_mask)
np.savetxt(output_dir / f"non_target_mask_{n_train}.txt", non_target_mask)
np.savetxt(output_dir / f"false_mask_{n_train}.txt", false_mask)

print(f"Data saved to {output_dir}")

# Plotting functions
def plot_gnn_vs_filter_scores_matched(filter_scores, gnn_scores, target_mask, non_target_mask, false_mask):
    """Plot GNN scores vs Filter scores for matched edges using 2D histogram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create 2D histogram for all data
    hist, xedges, yedges = np.histogram2d(filter_scores, gnn_scores, bins=50, range=[[0, 1], [0, 1]])
    
    # Plot 2D histogram
    im = ax.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Number of Edges')
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2, label='y=x reference')
    
    # Calculate and plot binned averages for each edge type
    edge_types = [
        ("Target", target_mask, 'tab:blue'),
        ("Non-target", non_target_mask, 'tab:green'),
        ("False", false_mask, 'tab:orange')
    ]
    
    for edge_type, mask, color in edge_types:
        if np.sum(mask) > 0:
            # Create bins and calculate means
            bins = np.linspace(0, 1, 21)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_indices = np.digitize(filter_scores[mask], bins)
            
            bin_means = []
            bin_stds = []
            for i in range(1, len(bins)):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 5:
                    bin_means.append(np.mean(gnn_scores[mask][bin_mask]))
                    bin_stds.append(np.std(gnn_scores[mask][bin_mask]) / np.sqrt(np.sum(bin_mask)))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            # Plot binned averages
            valid_mask = ~np.isnan(bin_means)
            if np.sum(valid_mask) > 0:
                ax.errorbar(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                           yerr=np.array(bin_stds)[valid_mask],
                           color=color, marker='o', markersize=4, linewidth=2, 
                           label=f'{edge_type} (binned avg)', alpha=0.9)
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Score', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
        outside=True
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_scores_matched_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_scores_matched_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter scores plot saved to {save_path}")


def plot_gnn_vs_filter_scores_individual(filter_scores, gnn_scores, mask, label, color, cmap_color):
    """Plot GNN scores vs Filter scores for a specific edge type using binned averages."""
    if np.sum(mask) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create 2D histogram for background
    hist, xedges, yedges = np.histogram2d(filter_scores[mask], gnn_scores[mask], bins=30, range=[[0, 1], [0, 1]])
    im = ax.imshow(hist.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap=cmap_color, alpha=0.7)
    plt.colorbar(im, ax=ax, label='Number of Edges')
    
    # Calculate and plot binned averages
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(filter_scores[mask], bins)
    
    bin_means = []
    bin_stds = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 5:
            bin_means.append(np.mean(gnn_scores[mask][bin_mask]))
            bin_stds.append(np.std(gnn_scores[mask][bin_mask]) / np.sqrt(np.sum(bin_mask)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    # Plot binned averages
    valid_mask = ~np.isnan(bin_means)
    if np.sum(valid_mask) > 0:
        ax.errorbar(bin_centers[valid_mask], np.array(bin_means)[valid_mask], 
                   yerr=np.array(bin_stds)[valid_mask],
                   color=color, marker='o', markersize=6, linewidth=3, 
                   label=f'{label} (binned avg)', alpha=1.0)
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='y=x reference')
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Score', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched {label} Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
        outside=True
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_scores_{label.lower()}_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_scores_{label.lower()}_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter scores ({label}) plot saved to {save_path}")


def plot_gnn_vs_filter_scores_matched_log(filter_scores, gnn_scores, target_mask, non_target_mask, false_mask):
    """Plot GNN scores vs Filter scores for matched edges with loglog scale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter out zero values for log scale
    nonzero_mask = (filter_scores > 0) & (gnn_scores > 0)
    
    # Create 2D histogram for all data (log scale)
    filter_log = filter_scores[nonzero_mask]
    gnn_log = gnn_scores[nonzero_mask]
    
    hist, xedges, yedges = np.histogram2d(np.log10(filter_log), np.log10(gnn_log), bins=50)
    im = ax.imshow(hist.T, origin='lower', extent=[np.log10(np.min(filter_log)), 0, 
                                                   np.log10(np.min(gnn_log)), 0], 
                   aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Number of Edges')
    
    # Add diagonal line in log space
    min_val = max(np.min(filter_log), np.min(gnn_log))
    ax.plot([np.log10(min_val), 0], [np.log10(min_val), 0], 'r--', alpha=0.8, linewidth=2, label='y=x reference')
    
    # Calculate and plot binned averages for each edge type in log space
    edge_types = [
        ("Target", target_mask, 'tab:blue'),
        ("Non-target", non_target_mask, 'tab:green'),
        ("False", false_mask, 'tab:orange')
    ]
    
    for edge_type, mask, color in edge_types:
        combined_mask = mask & nonzero_mask
        if np.sum(combined_mask) > 0:
            filter_masked = filter_scores[combined_mask]
            gnn_masked = gnn_scores[combined_mask]
            
            # Create log-spaced bins
            log_bins = np.logspace(np.log10(np.min(filter_masked)), 0, 21)
            bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])  # Geometric mean
            bin_indices = np.digitize(filter_masked, log_bins)
            
            bin_means = []
            bin_stds = []
            for i in range(1, len(log_bins)):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 5:
                    bin_means.append(np.mean(gnn_masked[bin_mask]))
                    bin_stds.append(np.std(gnn_masked[bin_mask]) / np.sqrt(np.sum(bin_mask)))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            # Plot binned averages in log space
            valid_mask = ~np.isnan(bin_means) & (np.array(bin_means) > 0)
            if np.sum(valid_mask) > 0:
                ax.errorbar(np.log10(bin_centers[valid_mask]), np.log10(np.array(bin_means)[valid_mask]), 
                           yerr=np.array(bin_stds)[valid_mask] / (np.array(bin_means)[valid_mask] * np.log(10)),
                           color=color, marker='o', markersize=4, linewidth=2, 
                           label=f'{edge_type} (binned avg)', alpha=0.9)
    
    ax.set_xlabel(r'$\log_{10}$(Filter Score)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel(r'$\log_{10}$(GNN Score)', fontsize=14, ha="right", y=0.95)
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
        outside=True
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_scores_matched_loglog_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_scores_matched_loglog_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter scores (loglog scale) plot saved to {save_path}")


def plot_gnn_vs_filter_scores_individual_log(filter_scores, gnn_scores, mask, label, color, cmap_color):
    """Plot GNN scores vs Filter scores for a specific edge type with loglog scale."""
    if np.sum(mask) == 0:
        return
        
    # Filter out zero values for log scale
    nonzero_mask = (filter_scores[mask] > 0) & (gnn_scores[mask] > 0)
    if np.sum(nonzero_mask) == 0:
        return
    
    filter_masked = filter_scores[mask][nonzero_mask]
    gnn_masked = gnn_scores[mask][nonzero_mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create 2D histogram for background (log scale)
    hist, xedges, yedges = np.histogram2d(np.log10(filter_masked), np.log10(gnn_masked), bins=30)
    im = ax.imshow(hist.T, origin='lower', 
                   extent=[np.log10(np.min(filter_masked)), np.log10(np.max(filter_masked)),
                          np.log10(np.min(gnn_masked)), np.log10(np.max(gnn_masked))], 
                   aspect='auto', cmap=cmap_color, alpha=0.7)
    plt.colorbar(im, ax=ax, label='Number of Edges')
    
    # Calculate and plot binned averages in log space
    log_bins = np.logspace(np.log10(np.min(filter_masked)), np.log10(np.max(filter_masked)), 21)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])  # Geometric mean
    bin_indices = np.digitize(filter_masked, log_bins)
    
    bin_means = []
    bin_stds = []
    for i in range(1, len(log_bins)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 5:
            bin_means.append(np.mean(gnn_masked[bin_mask]))
            bin_stds.append(np.std(gnn_masked[bin_mask]) / np.sqrt(np.sum(bin_mask)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    # Plot binned averages in log space
    valid_mask = ~np.isnan(bin_means) & (np.array(bin_means) > 0)
    if np.sum(valid_mask) > 0:
        ax.errorbar(np.log10(bin_centers[valid_mask]), np.log10(np.array(bin_means)[valid_mask]), 
                   yerr=np.array(bin_stds)[valid_mask] / (np.array(bin_means)[valid_mask] * np.log(10)),
                   color=color, marker='o', markersize=6, linewidth=3, 
                   label=f'{label} (binned avg)', alpha=1.0)
    
    # Add diagonal line in log space
    min_val = max(np.min(filter_masked), np.min(gnn_masked))
    max_val = min(np.max(filter_masked), np.max(gnn_masked))
    ax.plot([np.log10(min_val), np.log10(max_val)], [np.log10(min_val), np.log10(max_val)], 
            'k--', alpha=0.5, linewidth=1, label='y=x reference')
    ax.set_xlabel(r'$\log_{10}$(Filter Score)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel(r'$\log_{10}$(GNN Score)', fontsize=14, ha="right", y=0.95)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched {label} Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
        outside=True
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_scores_{label.lower()}_loglog_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_scores_{label.lower()}_loglog_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter scores ({label}, loglog scale) plot saved to {save_path}")


def plot_gnn_vs_filter_uncertainties_matched(filter_uncertainties, gnn_uncertainties, target_mask, non_target_mask, false_mask):
    """Plot GNN uncertainties vs Filter uncertainties for matched edges."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot different edge types
    if np.sum(target_mask) > 0:
        ax.scatter(filter_uncertainties[target_mask], gnn_uncertainties[target_mask], 
                  alpha=0.6, s=0.1, color='tab:blue', label='Target Edges')
    
    if np.sum(non_target_mask) > 0:
        ax.scatter(filter_uncertainties[non_target_mask], gnn_uncertainties[non_target_mask], 
                  alpha=0.6, s=0.1, color='tab:green', label='Non-target Edges')
    
    if np.sum(false_mask) > 0:
        ax.scatter(filter_uncertainties[false_mask], gnn_uncertainties[false_mask], 
                  alpha=0.6, s=0.1, color='tab:orange', label='False Edges')
    
    # Add diagonal line
    # max_uncertainty = max(np.max(filter_uncertainties), np.max(gnn_uncertainties))
    max_uncertainty = max(np.max(filter_uncertainties[target_mask]), 
                          np.max(gnn_uncertainties[target_mask]),
                          np.max(filter_uncertainties[non_target_mask]),
                          np.max(gnn_uncertainties[non_target_mask]),
                          np.max(filter_uncertainties[false_mask]),
                          np.max(gnn_uncertainties[false_mask]))
    ax.plot([0, max_uncertainty], [0, max_uncertainty], 'k--', alpha=0.5, linewidth=1, label='y=x reference')
    
    ax.set_xlabel('Filter Uncertainty (Std Dev)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, max_uncertainty)
    ax.set_ylim(0, max_uncertainty)
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_uncertainties_matched_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_uncertainties_matched_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter uncertainties plot saved to {save_path}")


def plot_gnn_vs_filter_uncertainties_individual(filter_uncertainties, gnn_uncertainties, mask, label, color):
    """Plot GNN uncertainties vs Filter uncertainties for a specific edge type."""
    if np.sum(mask) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(filter_uncertainties[mask], gnn_uncertainties[mask], 
              alpha=0.6, s=0.1, color=color, label=f'{label} Edges')
    
    # Add diagonal line
    max_uncertainty = max(np.max(filter_uncertainties[mask]), np.max(gnn_uncertainties[mask]))
    ax.plot([0, max_uncertainty], [0, max_uncertainty], 'k--', alpha=0.5, linewidth=1, label='y=x reference')
    
    ax.set_xlabel('Filter Uncertainty (Std Dev)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, max_uncertainty)
    ax.set_ylim(0, max_uncertainty)
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched {label} Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_uncertainties_{label.lower()}_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_uncertainties_{label.lower()}_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter uncertainties ({label}) plot saved to {save_path}")


def plot_gnn_vs_filter_uncertainties_matched_log(filter_uncertainties, gnn_uncertainties, target_mask, non_target_mask, false_mask):
    """Plot GNN uncertainties vs Filter uncertainties for matched edges with loglog scale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter out zero/negative values for log scale
    nonzero_mask = (filter_uncertainties > 0) & (gnn_uncertainties > 0)
    
    if np.sum(nonzero_mask) == 0:
        print("No positive uncertainties found for log scale plot")
        return
    
    # Get the full range from all truth labels to ensure heatmap covers everything
    all_combined_masks = []
    edge_types = [
        ("Target", target_mask, 'tab:blue'),
        ("Non-target", non_target_mask, 'tab:green'),
        ("False", false_mask, 'tab:orange')
    ]
    
    for edge_type, mask, color in edge_types:
        combined_mask = mask & nonzero_mask
        if np.sum(combined_mask) > 0:
            all_combined_masks.append(combined_mask)
    
    # Combine all masks to get the full range
    if all_combined_masks:
        full_combined_mask = np.zeros_like(nonzero_mask, dtype=bool)
        for combined_mask in all_combined_masks:
            full_combined_mask = full_combined_mask | combined_mask
        
        # Use the full combined mask for determining the range
        filter_log = filter_uncertainties[full_combined_mask]
        gnn_log = gnn_uncertainties[full_combined_mask]
    else:
        # Fallback to original nonzero_mask if no truth labels have data
        filter_log = filter_uncertainties[nonzero_mask]
        gnn_log = gnn_uncertainties[nonzero_mask]
    
    # Create 2D histogram for all data (log scale) using the full range
    hist, xedges, yedges = np.histogram2d(np.log10(filter_log), np.log10(gnn_log), bins=60)
    im = ax.imshow(hist.T, origin='lower', extent=[np.log10(np.min(filter_log)), np.log10(np.max(filter_log)), 
                                                   np.log10(np.min(gnn_log)), np.log10(np.max(gnn_log))], 
                   aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Number of Edges')
    
    # Add diagonal line in log space
    min_val = max(np.min(filter_log), np.min(gnn_log))
    max_val = min(np.max(filter_log), np.max(gnn_log))
    ax.plot([np.log10(min_val), np.log10(max_val)], [np.log10(min_val), np.log10(max_val)], 
            'r--', alpha=0.8, linewidth=2, label='y=x reference')
    
    # Calculate and plot binned averages for each edge type in log space  
    for edge_type, mask, color in edge_types:
        combined_mask = mask & nonzero_mask
        if np.sum(combined_mask) > 0:
            filter_masked = filter_uncertainties[combined_mask]
            gnn_masked = gnn_uncertainties[combined_mask]
            
            # Create log-spaced bins
            log_bins = np.logspace(np.log10(np.min(filter_masked)), np.log10(np.max(filter_masked)), 21)
            bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])  # Geometric mean
            bin_indices = np.digitize(filter_masked, log_bins)
            
            bin_means = []
            bin_stds = []
            for i in range(1, len(log_bins)):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 1:
                    bin_means.append(np.mean(gnn_masked[bin_mask]))
                    bin_stds.append(np.std(gnn_masked[bin_mask]) / np.sqrt(np.sum(bin_mask)))
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            # Plot binned averages in log space
            valid_mask = ~np.isnan(bin_means) & (np.array(bin_means) > 0)
            if np.sum(valid_mask) > 0:
                ax.errorbar(np.log10(bin_centers[valid_mask]), np.log10(np.array(bin_means)[valid_mask]), 
                           yerr=np.array(bin_stds)[valid_mask] / (np.array(bin_means)[valid_mask] * np.log(10)),
                           color=color, marker='o', markersize=4, linewidth=2, 
                           label=f'{edge_type} (binned avg)', alpha=0.9)
    
    ax.set_xlabel(r'$\log_{10}$(Filter Uncertainty)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel(r'$\log_{10}$(GNN Uncertainty)', fontsize=14, ha="right", y=0.95)
    ax.legend(loc='lower left', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
        outside=True
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_vs_filter_uncertainties_matched_loglog_{n_train}.png"
    save_path_svg = output_dir / f"gnn_vs_filter_uncertainties_matched_loglog_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN vs Filter uncertainties (loglog scale) plot saved to {save_path}")


def plot_gnn_uncertainty_vs_filter_scores_matched(filter_scores, gnn_uncertainties, target_mask, non_target_mask, false_mask):
    """Plot GNN uncertainties vs Filter scores for matched edges."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot different edge types
    if np.sum(target_mask) > 0:
        ax.scatter(filter_scores[target_mask], gnn_uncertainties[target_mask], 
                  alpha=0.6, s=0.1, color='tab:blue', label='Target Edges')
    
    if np.sum(non_target_mask) > 0:
        ax.scatter(filter_scores[non_target_mask], gnn_uncertainties[non_target_mask], 
                  alpha=0.6, s=0.1, color='tab:green', label='Non-target Edges')
    
    if np.sum(false_mask) > 0:
        ax.scatter(filter_scores[false_mask], gnn_uncertainties[false_mask], 
                  alpha=0.6, s=0.1, color='tab:orange', label='False Edges')
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(gnn_uncertainties))
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_uncertainty_vs_filter_scores_matched_{n_train}.png"
    save_path_svg = output_dir / f"gnn_uncertainty_vs_filter_scores_matched_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN uncertainty vs Filter scores plot saved to {save_path}")


def plot_gnn_uncertainty_vs_filter_scores_individual(filter_scores, gnn_uncertainties, mask, label, color):
    """Plot GNN uncertainties vs Filter scores for a specific edge type."""
    if np.sum(mask) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(filter_scores[mask], gnn_uncertainties[mask], 
              alpha=0.6, s=0.1, color=color, label=f'{label} Edges')
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(gnn_uncertainties[mask]))
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched {label} Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_uncertainty_vs_filter_scores_{label.lower()}_{n_train}.png"
    save_path_svg = output_dir / f"gnn_uncertainty_vs_filter_scores_{label.lower()}_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN uncertainty vs Filter scores ({label}) plot saved to {save_path}")


def plot_gnn_uncertainty_vs_filter_scores_matched_log(filter_scores, gnn_uncertainties, target_mask, non_target_mask, false_mask):
    """Plot GNN uncertainties vs Filter scores for matched edges with log scale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot different edge types
    if np.sum(target_mask) > 0:
        ax.scatter(filter_scores[target_mask], gnn_uncertainties[target_mask], 
                  alpha=0.6, s=0.1, color='tab:blue', label='Target Edges')
    
    if np.sum(non_target_mask) > 0:
        ax.scatter(filter_scores[non_target_mask], gnn_uncertainties[non_target_mask], 
                  alpha=0.6, s=0.1, color='tab:green', label='Non-target Edges')
    
    if np.sum(false_mask) > 0:
        ax.scatter(filter_scores[false_mask], gnn_uncertainties[false_mask], 
                  alpha=0.6, s=0.1, color='tab:orange', label='False Edges')
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(gnn_uncertainties))
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_uncertainty_vs_filter_scores_matched_log_{n_train}.png"
    save_path_svg = output_dir / f"gnn_uncertainty_vs_filter_scores_matched_log_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN uncertainty vs Filter scores (log scale) plot saved to {save_path}")


def plot_gnn_uncertainty_vs_filter_scores_individual_log(filter_scores, gnn_uncertainties, mask, label, color):
    """Plot GNN uncertainties vs Filter scores for a specific edge type with log scale."""
    if np.sum(mask) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(filter_scores[mask], gnn_uncertainties[mask], 
              alpha=0.6, s=0.1, color=color, label=f'{label} Edges')
    
    ax.set_xlabel('Filter Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('GNN Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(gnn_uncertainties[mask]))
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=12)
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        + f"Matched {label} Edges (Filter cut ≥ {filter_score_cut})" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: 0.1" + "\n",
    )
    
    fig.tight_layout()
    
    save_path = output_dir / f"gnn_uncertainty_vs_filter_scores_{label.lower()}_log_{n_train}.png"
    save_path_svg = output_dir / f"gnn_uncertainty_vs_filter_scores_{label.lower()}_log_{n_train}.svg"
    
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"GNN uncertainty vs Filter scores ({label}, log scale) plot saved to {save_path}")


# Create all plots
print("Creating plots...")

# Combined plots
plot_gnn_vs_filter_scores_matched(filter_mean_scores, gnn_mean_scores, target_mask, non_target_mask, false_mask)
plot_gnn_vs_filter_uncertainties_matched(filter_uncertainties, gnn_uncertainties, target_mask, non_target_mask, false_mask)
plot_gnn_uncertainty_vs_filter_scores_matched(filter_mean_scores, gnn_uncertainties, target_mask, non_target_mask, false_mask)

# Individual edge type plots
edge_types = [
    ("Target", target_mask, 'tab:blue', 'Blues'),
    ("Non-target", non_target_mask, 'tab:green', 'Greens'),
    ("False", false_mask, 'tab:orange', 'Oranges')
]

for label, mask, color, cmap_color in edge_types:
    plot_gnn_vs_filter_scores_individual(filter_mean_scores, gnn_mean_scores, mask, label, color, cmap_color)
    plot_gnn_vs_filter_uncertainties_individual(filter_uncertainties, gnn_uncertainties, mask, label, color)
    plot_gnn_uncertainty_vs_filter_scores_individual(filter_mean_scores, gnn_uncertainties, mask, label, color)

# Log scale plots - combined
plot_gnn_vs_filter_scores_matched_log(filter_mean_scores, gnn_mean_scores, target_mask, non_target_mask, false_mask)
plot_gnn_vs_filter_uncertainties_matched_log(filter_uncertainties, gnn_uncertainties, target_mask, non_target_mask, false_mask)
plot_gnn_uncertainty_vs_filter_scores_matched_log(filter_mean_scores, gnn_uncertainties, target_mask, non_target_mask, false_mask)

# Log scale plots - individual
for label, mask, color, cmap_color in edge_types:
    plot_gnn_vs_filter_scores_individual_log(filter_mean_scores, gnn_mean_scores, mask, label, color, cmap_color)
    # plot_gnn_vs_filter_uncertainties_individual_log(filter_uncertainties, gnn_uncertainties, mask, label, color, cmap_color)
    plot_gnn_uncertainty_vs_filter_scores_individual_log(filter_mean_scores, gnn_uncertainties, mask, label, color)

print("Algorithm completed successfully!")
print(f"Total matched edges analyzed: {len(matched_edges)}")
print(f"Target edges: {np.sum(target_mask)}")
print(f"Non-target edges: {np.sum(non_target_mask)}")
print(f"False edges: {np.sum(false_mask)}")

