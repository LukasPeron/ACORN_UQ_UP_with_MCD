import torch
import matplotlib.pyplot as plt
import numpy as np
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
from pathlib import Path
import os
import tqdm
# def find_edge_indices(graph_edges, edges_prop):
#     """
#     Returns a 1D array containing the position indices of edges_prop in graph_edges.
    
#     Parameters:
#     - graph_edges: numpy array of shape (2, n_edges) representing all edges
#     - edges_prop: numpy array of shape (2, m) representing a subset of edges
    
#     Returns:
#     - indices: numpy array of shape (m,) containing position indices
#     """
#     # Create a dictionary mapping edge pairs to their indices
#     # Use frozenset to handle undirected edges (same hash regardless of order)
#     edge_to_idx = {frozenset([a, b]): i for i, (a, b) in enumerate(zip(graph_edges[0], graph_edges[1]))}
    
#     # Find indices using dictionary lookup in a vectorized manner
#     indices = np.array([edge_to_idx.get(frozenset([a, b]), 0) for a, b in zip(edges_prop[0], edges_prop[1])])
    
#     return indices


# def plot_edges_score_vs_pt(all_flat_pt, all_flat_scores_track):
#     """
#     Plot mean edge scores vs. pT with error bands for all track edges.
#     Creates a single plot for all edges without distinguishing between target and non-target.
#     """
    
#     # Define PT bins based on units
#     pt_min = 1
#     pt_max = 50
    
#     # Create bins for PT (log scale)
#     pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 20)
    
#     # Use all edges without filtering by truth type
#     edge_pt = all_flat_pt
#     edge_scores = all_flat_scores_track
    
#     if len(edge_pt) == 0:  # Skip if no edges
#         return
        
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     pt_indices = np.digitize(edge_pt, pt_bins) - 1
    
#     mean_scores = []
#     bin_centers = []
#     score_errors = []
    
#     for i in range(len(pt_bins) - 1):
#         bin_mask = (pt_indices == i)
        
#         if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
#             scores_in_bin = edge_scores[bin_mask]
#             mean_score = np.mean(scores_in_bin)
#             error = np.std(scores_in_bin) / np.sqrt(np.sum(bin_mask))
            
#             # Geometric mean for log scale
#             bin_center = np.sqrt(pt_bins[i] * pt_bins[i+1])
            
#             mean_scores.append(mean_score)
#             score_errors.append(error)
#             bin_centers.append(bin_center)
    
#     # Plot line for all edges
#     if mean_scores:
#         ax.plot(bin_centers, mean_scores, '-o', linewidth=2, color='red', label='All Track Edges')
#         ax.fill_between(
#             bin_centers, 
#             np.array(mean_scores) - np.array(score_errors), 
#             np.array(mean_scores) + np.array(score_errors), 
#             alpha=0.3,
#             color='red',
#             edgecolor=None
#         )
    
#     # Configure axis
#     ax.set_xscale('log')
#     ax.set_xlabel(f'$p_T$ [GeV]', fontsize=14, ha="right", x=0.95)
#     ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
    
#     # Set y-axis limits to the range of scores
#     ax.set_ylim(0, 1)
    
#     # Set x-axis limits
#     ax.set_xlim(pt_min, pt_max)
    
#     # Add legend
#     ax.legend(loc='upper right', fontsize=14)
        
#     # Apply ATLAS styling
#     atlasify(f"1400 train events",
#         r"$ | \eta | < 4$" + "\n"
#         + f"Evaluated on 50 events in valset"
#     )
    
#     fig.tight_layout()
#     # Save the figure
#     fig.savefig("all_track_edges_score_vs_pt_cal.svg")
#     plt.close(fig)

# pt_list = []
# score_list = []

# for event_id in tqdm.tqdm(range(21000, 22755+1)):
#     results_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/valset"
#     if list(Path(f"{results_path}/").glob(f"*0000{event_id}*"))!=[]:   
#         post_step = list(Path(f"{results_path}/").glob(f"*0000{event_id}*"))[0]
#         event = torch.load(post_step, map_location="cpu", weights_only=False)
#         pt_list.extend(event.pt.cpu().numpy())
#         track_mask = find_edge_indices(event.edge_index, event.track_edges)
#         scores = event.scores.cpu().numpy()[track_mask]
#         score_list.extend(scores)

# # Convert pt_list to a numpy array
# pt_list = np.array(pt_list)/1000.  # Convert to GeV

# # Convert score_list to a numpy array
# score_list = np.array(score_list)

# np.savetxt("cal_scores.txt", score_list)
# np.savetxt("cal_pt.txt", pt_list)

# # Plot the scores vs. pT
# plot_edges_score_vs_pt(pt_list, score_list)

def plot_edges_score_vs_pt_comparison(pt_uncal, score_uncal, pt_cal, score_cal):
    """
    Plot residual (uncalibrated - calibrated) edge scores vs. pT with error bands.
    Creates a plot showing the difference between uncalibrated and calibrated scores.
    
    Parameters:
    - pt_uncal: array of pT values for uncalibrated scores
    - score_uncal: array of uncalibrated edge scores
    - pt_cal: array of pT values for calibrated scores  
    - score_cal: array of calibrated edge scores
    """
    
    # Define PT bins based on units
    pt_min = 1
    pt_max = 50
    
    # Create bins for PT (log scale)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 20)
    
    # Assuming pt arrays are the same, use pt_uncal for binning
    edge_pt = pt_uncal
    residual_scores = score_uncal - score_cal
    
    if len(edge_pt) == 0:  # Skip if no edges
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pt_indices = np.digitize(edge_pt, pt_bins) - 1
    
    mean_residuals = []
    bin_centers = []
    residual_errors = []
    
    for i in range(len(pt_bins) - 1):
        bin_mask = (pt_indices == i)
        
        if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
            residuals_in_bin = residual_scores[bin_mask]
            mean_residual = np.mean(residuals_in_bin)
            error = np.std(residuals_in_bin) / np.sqrt(np.sum(bin_mask))
            
            # Geometric mean for log scale
            bin_center = np.sqrt(pt_bins[i] * pt_bins[i+1])
            
            mean_residuals.append(mean_residual)
            residual_errors.append(error)
            bin_centers.append(bin_center)
    
    # Plot residual line
    if mean_residuals:
        ax.plot(bin_centers, mean_residuals, '-o', linewidth=2, color='blue', label='Uncalibrated - Calibrated')
        ax.fill_between(
            bin_centers, 
            np.array(mean_residuals) - np.array(residual_errors), 
            np.array(mean_residuals) + np.array(residual_errors), 
            alpha=0.3,
            color='blue',
            edgecolor=None
        )
    
    # Configure axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'$p_T$ [GeV]', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Score Residual (Uncal - Cal)', fontsize=14, ha="right", y=0.95)
    
    # Set x-axis limits
    ax.set_xlim(pt_min, pt_max)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
        
    # Apply ATLAS styling
    atlasify(f"1400 train events",
        r"$ | \eta | < 4$" + "\n"
        + f"Evaluated on 50 events in valset"
    )
    
    fig.tight_layout()
    # Save the figure
    fig.savefig("score_residual_vs_pt.svg")
    plt.close(fig)

pt_cal = np.loadtxt("cal_pt.txt")
score_cal = np.loadtxt("cal_scores.txt")
pt_uncal = np.loadtxt("uncal_pt.txt")
score_uncal = np.loadtxt("uncal_scores.txt")

# Plot the scores vs. pT for both calibrated and uncalibrated scores
plot_edges_score_vs_pt_comparison(pt_uncal, score_uncal, pt_cal, score_cal)