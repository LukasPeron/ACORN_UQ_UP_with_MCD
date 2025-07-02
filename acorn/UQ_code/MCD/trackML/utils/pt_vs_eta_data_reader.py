import torch 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import tqdm

set = "trainset"
eta = []
pt = []
# Loop over the event IDs
for event_id in tqdm.tqdm(range(21000, 22755+1)):
    results_path = "/pscratch/sd/l/lperon/ATLAS/data_dir/Example_3/feature_store/"
    if list(Path(f"{results_path}/{set}/").glob(f"*0000{event_id}*.pyg"))!=[]:   
        post_step = list(Path(f"{results_path}/{set}/").glob(f"*0000{event_id}*.pyg"))[0]
        data_post_step = torch.load(post_step)
        eta.append((data_post_step.eta[data_post_step.track_edges[0]]).numpy())
        pt.append(data_post_step.pt.numpy())

# Convert lists to numpy arrays
eta = np.concatenate(eta)
pt = np.concatenate(pt)/1000  # Convert pt to GeV

# Plot histogram pt vs eta
plt.figure(figsize=(10, 6))
plt.hist2d(eta, pt, bins=[100, 100], cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
plt.ylabel(r'$p_T$ [GeV]', fontsize=14, ha="right", y=0.95)
plt.yscale('log')
atlasify(f"Data reader {set}", outside=True)
plt.tight_layout()
plt.savefig(f'pt_vs_eta_{set}.svg')
plt.close()