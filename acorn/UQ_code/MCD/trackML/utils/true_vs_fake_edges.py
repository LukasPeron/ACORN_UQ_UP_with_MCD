import torch 
from torch_geometric.data import Data
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from acorn.utils.eval_utils import find_edge_indices

nb_true_edges = []
nb_total_edges = []
# low_edges_scores_pt = []
# low_edges_scores_eta = []
# Loop over the event IDs
scores_full = []
for step in ["filter"]:
    set = "valset"
    nb_edge = 0
    for event_id in tqdm.tqdm(range(21000, 22755+1)):
        results_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/"
        if list(Path(f"{results_path}/{step}/{set}/").glob(f"*0000{event_id}*"))!=[]:   
            post_step = list(Path(f"{results_path}/{step}/{set}/").glob(f"*0000{event_id}*"))[0]
            data_post_step = torch.load(post_step, map_location="cpu")
            # track_mask = find_edge_indices(data_post_step.edge_index.cpu().numpy(), data_post_step.track_edges.cpu().numpy())
            # low_scores_edges = data_post_step.scores < 0.05
            # low_scores_edges_track = data_post_step.scores[track_mask] < 0.05
            # true_target_edges = (data_post_step.edge_weights == 2) & (data_post_step.y == 1.0)
            # true_target_edges_track = (data_post_step.edge_weights[track_mask] == 2)
            # nb_edge += (low_scores_edges & true_target_edges).sum()
            # low_edges_scores_pt.append(data_post_step.pt[low_scores_edges_track & true_target_edges_track].cpu().numpy())
            # lst_eta = data_post_step.track_edges[0][low_scores_edges_track & true_target_edges_track].cpu().numpy()
            # low_edges_scores_eta.append(data_post_step.eta[lst_eta].cpu().numpy())

            graph_size = data_post_step.edge_index.shape[1]
            # true_edges = (data_post_step.y).sum()
            # nb_true_edges.append(true_edges)
            nb_total_edges.append(graph_size)
            scores = data_post_step.scores
            scores_full.append(scores.cpu().numpy())

scores_full = np.concatenate(scores_full)
print("score max:", scores_full.max())

# flatten the lists of scores
# low_edges_scores_pt = np.concatenate(low_edges_scores_pt)
# low_edges_scores_eta = np.concatenate(low_edges_scores_eta)

# #plot histograms of low scores edges pt and eta
# print(nb_edge)
# plt.figure(figsize=(10, 6))
# plt.hist(low_edges_scores_pt, bins=50, histtype='step', label='Low Scores Edges pT')
# #use x log scale
# plt.xscale('log')
# plt.savefig(f'low_scores_edges_pt_{set}.svg')

# plt.figure(figsize=(10, 6))
# plt.hist(low_edges_scores_eta, bins=50, histtype="step", label='Low Scores Edges eta')
# plt.savefig(f'low_scores_edges_eta_{set}.svg')

# Plot the graph sizes
# plt.figure(figsize=(10, 6))
# plt.plot(nb_true_edges, label='True Edges', marker='o')
# plt.plot(nb_total_edges, label='Total Edges', marker='o')
# plt.title(f'Number of Edges at Different Stages from {set} ({len(nb_true_edges)} graphs)')
# plt.xlabel('Event ID (arbitrary)')
# plt.ylabel('Number of Edges')
# plt.yscale('log')
# plt.legend()
# plt.savefig(f'graph_sizes_{set}.svg')