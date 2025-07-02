import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import torch
from pathlib import Path
import tqdm
import os
from acorn.utils.eval_utils import plot_edge_scores_distribution
"""
Plot cumulative histograms of the edge scores of GNN outputs
histogram shows on the x axis the score threshold and on the y axis the fraction of edges with score below that threshold
"""

calib = False
truth_target = []
truth_non_target = []
false = []
scores = []

data_path = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/valset")
for event_id in tqdm.tqdm(range(21000, 22755+1)):
    if list(data_path.glob(f"*0000{event_id}*"))!=[]:   
        post_step = list(data_path.glob(f"*0000{event_id}*"))[0]
        data_post_step = torch.load(post_step, map_location="cpu")
        scores.append(data_post_step.scores.cpu().numpy())
        truth_target.append(((data_post_step.edge_weights == 1.0) & (data_post_step.y == 1.0)).cpu().numpy())
        truth_non_target.append(((data_post_step.edge_weights == 0.0) & (data_post_step.y == 1.0)).cpu().numpy())
        false.append((data_post_step.y == 0.0).cpu().numpy())

# flatten the lists of scores
scores = np.concatenate(scores)
truth_target = np.concatenate(truth_target)
truth_non_target = np.concatenate(truth_non_target)
false = np.concatenate(false)
n_target = np.sum(truth_target)
n_non_target = np.sum(truth_non_target)
n_false = np.sum(false)

truth_target_scores = scores[truth_target]
truth_non_target_scores = scores[truth_non_target]
false_scores = scores[false]

# Define bins for the histogram
bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
# Compute histograms
hist_target, _ = np.histogram(truth_target_scores, bins=bins)
hist_non_target, _ = np.histogram(truth_non_target_scores, bins=bins)
hist_false, _ = np.histogram(false_scores, bins=bins)
# Compute cumulative sums
cum_target = np.cumsum(hist_target) / len(truth_target_scores)
cum_non_target = np.cumsum(hist_non_target) / len(truth_non_target_scores)
cum_false = np.cumsum(hist_false) / len(false_scores)
# Plot cumulative histograms
plt.figure(figsize=(8, 6))
plt.plot(bins[:-1], cum_target, label='True Target Edges', drawstyle='steps-post', color="tab:blue", linewidth=2)
plt.plot(bins[:-1], cum_non_target, label='True Non-Target Edges', drawstyle='steps-post', color="tab:green", linewidth=2)
plt.plot(bins[:-1], cum_false, label='False Edges', drawstyle='steps-post', color="tab:orange", linewidth=2)
plt.xlabel('Edge Score Threshold', ha='right', fontsize=14, x=0.95)
plt.ylabel('Cumulative Fraction of Edges', ha='right', fontsize=14, y=0.95)
plt.legend(fontsize=12)
atlasify(f"1400 train events",
        r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
        f"{'Calibrated' if calib else 'Uncalibrated'} GNN scores",
    )
plt.tight_layout()
plt.savefig(f'cumulative_histogram_scores_{"calib" if calib else "uncalib"}.svg')

config = {
    "n_train": 1400,
    "target_tracks": {"track_particle_pt": [1000]},
    "dataset_size": 50,
    "stage_dir": "",
    "dataset": "valset",
    "score_cut": 0.5
    }

# plot_edge_scores_distribution(all_flat_scores=scores, all_flat_target_truth=truth_target, all_flat_non_target_truth=truth_non_target, all_flat_false=false, dataset="valset", config=config, plot_config={}, dropout_value=0, UQ_propagation=False):

fig, ax = plt.subplots(figsize=(8, 6))
    
# Plot all three edge types on the same plot
for edge_type, truth_mask, color, hist_integral in [
    ("Target", truth_target, 'tab:blue', n_target),
    ("Non-target", truth_non_target, 'tab:green', n_non_target),
    ("False", false, 'tab:orange', n_false)
]:
    edge_scores = scores[truth_mask]
    if len(edge_scores) > 0:
        ax.hist(edge_scores, bins=100, histtype='step', label=f'{edge_type} Edges - {hist_integral} edges', 
                linewidth=2, color=color)

ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
ax.set_ylabel('Count', fontsize=14, ha="right", y=0.95)
ax.set_yscale('log')
ax.legend(fontsize=14)
ax.set_xlim([0, 1])

score_cut = config["score_cut"]
ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7,
            label=f'Score Cut ({score_cut})')

atlasify(f"1400 train events",
    r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
    r"Edge score cut: " + str(score_cut) + "\n"
    + f"Evaluated on {config.get('dataset_size', 50)} events in valset" + "\n"
)

fig.tight_layout()
save_path_combined_svg = os.path.join(
    config["stage_dir"], 
    f"edge_scores_distribution_combined_{'calibrated' if calib else 'uncalibrated'}.svg"
)
fig.savefig(save_path_combined_svg)


fig, ax = plt.subplots(figsize=(8, 6))
    
# Plot all three edge types on the same plot
for edge_type, truth_mask, color, hist_integral in [
    ("Target", truth_target, 'tab:blue', n_target),
    ("Non-target", truth_non_target, 'tab:green', n_non_target),
    ("False", false, 'tab:orange', n_false)
]:
    edge_scores = scores[truth_mask]
    if len(edge_scores) > 0:
        ax.hist(edge_scores, bins=100, histtype='step', label=f'{edge_type} Edges - {hist_integral} edges', 
                linewidth=2, color=color, weights=np.ones_like(edge_scores) / len(edge_scores))

ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
ax.set_ylabel('Normalized count', fontsize=14, ha="right", y=0.95)
ax.set_yscale('log')
ax.legend(fontsize=14)
ax.set_xlim([0, 1])

score_cut = config["score_cut"]
ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7,
            label=f'Score Cut ({score_cut})')

atlasify(f"1400 train events",
    r"Target: $p_T > 1$ GeV, $ | \eta | < 4$" + "\n"
    r"Edge score cut: " + str(score_cut) + "\n"
    + f"Evaluated on {config.get('dataset_size', 50)} events in valset" + "\n"
)

fig.tight_layout()
save_path_combined_svg = os.path.join(
    config["stage_dir"], 
    f"edge_scores_distribution_normalized_combined_{'calibrated' if calib else 'uncalibrated'}.svg"
)
fig.savefig(save_path_combined_svg)