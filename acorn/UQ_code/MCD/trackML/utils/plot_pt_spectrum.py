import torch
import matplotlib.pyplot as plt
import numpy as np
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
from pathlib import Path
import tqdm

def find_edge_indices(graph_edges, edges_prop):
    """
    Returns a 1D array containing the position indices of edges_prop in graph_edges.
    
    Parameters:
    - graph_edges: numpy array of shape (2, n_edges) representing all edges
    - edges_prop: numpy array of shape (2, m) representing a subset of edges
    
    Returns:
    - indices: numpy array of shape (m,) containing position indices
    """
    # Create a dictionary mapping edge pairs to their indices
    # Use frozenset to handle undirected edges (same hash regardless of order)
    edge_to_idx = {frozenset([a, b]): i for i, (a, b) in enumerate(zip(graph_edges[0], graph_edges[1]))}
    
    # Find indices using dictionary lookup in a vectorized manner
    indices = np.array([edge_to_idx.get(frozenset([a, b]), 0) for a, b in zip(edges_prop[0], edges_prop[1])])
    
    return indices


pt_list = []

for event_id in tqdm.tqdm(range(21000, 22755+1)):
    results_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/valset"
    if list(Path(f"{results_path}/").glob(f"*0000{event_id}*"))!=[]:   
        post_step = list(Path(f"{results_path}/").glob(f"*0000{event_id}*"))[0]
        data_post_step = torch.load(post_step, map_location="cpu")
        pt_list.extend(data_post_step.pt.cpu().numpy())

# Convert pt_list to a numpy array
pt_list = np.array(pt_list)/1000.  # Convert to GeV
# Plot the pT spectrum
plt.figure(figsize=(8, 6))
plt.hist(pt_list, bins=100, linewidth=2, histtype='step')
plt.vlines(1, 0, 4e6, color='k', linestyle='--', label=r'$p_T = 1$ GeV')
plt.legend(fontsize=12)
# plt.hist(pt_list, bins=100, label='pT Spectrum', linewidth=2, histtype='step')
plt.xlabel('Transverse Momentum $p_T$ [GeV]', fontsize=14, ha="right", x=0.95)
plt.ylabel('Number of Entries', fontsize=14, ha="right", y=0.95)
plt.yscale('log')
atlasify(f"1400 train events",
        r"Target: $p_T>1$ GeV, $| \eta | < 4$" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
    )
plt.ylim(0, 4e7)
plt.tight_layout()
plt.savefig("pt_spectrum.svg")
plt.savefig("pt_spectrum.pdf")
