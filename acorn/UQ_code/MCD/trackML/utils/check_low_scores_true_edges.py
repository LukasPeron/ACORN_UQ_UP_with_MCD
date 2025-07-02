from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import os
from acorn.utils.eval_utils import find_edge_indices
import matplotlib.pyplot as plt

n_train = 1400
# output_dir = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/test/")

gnn_dataset_name = "valset" 

# Load GNN model
gnn_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/artifacts/"
if list(Path(gnn_ckpt_path).glob("best*"))!=[]:
    gnn_checkpoint = list(Path(gnn_ckpt_path).glob("best*"))[0]
    gnn_model = InteractionGNN.load_from_checkpoint(gnn_checkpoint)
    gnn_model.hparams["data_split"] = [0,50,0]
    gnn_model.setup(stage="predict")
    gnn_model.hparams["input_cut"] = 0.0 #! Need to set input_cut to 0.0 for UQ propagation in order to avoid varying graph input sizes to the GNN
    dataset = getattr(gnn_model, gnn_dataset_name)
    print(dataset.hparams["input_cut"])
    print(f"GNN model loaded from {gnn_checkpoint}")

all_target_truth = [None for _ in range(len(dataset))] 
all_non_target_truth = [None for _ in range(len(dataset))] 
all_false = [None for _ in range(len(dataset))]
target_pt = [None for _ in range(len(dataset))]
target_eta = [None for _ in range(len(dataset))]

all_scores = []
all_track_scores = []
all_target_truth_track = [None for _ in range(len(dataset))]
all_non_target_truth_track = [None for _ in range(len(dataset))]

for num_event, event in enumerate(tqdm(dataset)):
    with torch.inference_mode():
        gnn_model.eval()
        gnn_dict = gnn_model.shared_evaluation(event.to(gnn_model.device), 0)
        event_gnn = gnn_dict["batch"]

    after_gnn_scores = gnn_dict["batch"].edge_scores.cpu().numpy()
    all_scores.append(after_gnn_scores)
    # Calculate the entropy of the edge scores, useful for epistemic/aleatoric uncertainty

    # Handling of differents targets categories of edges
    if all_target_truth[num_event] is None:
        all_target_truth[num_event] = event_gnn.edge_y.cpu().numpy() & (event_gnn.edge_weights==1).cpu().numpy() #* select only true target edges (ie. with pT>1GeV and nhits>3)
        all_non_target_truth[num_event] = event_gnn.edge_y.cpu().numpy() & (event_gnn.edge_weights==0).cpu().numpy() #* select only true non-target edges
        all_false[num_event] = (~event_gnn.edge_y.cpu().numpy()) & (event_gnn.edge_weights==0.1).cpu().numpy() #* select only false edges
        
    edge_index = event_gnn.edge_index.cpu().numpy()
    track_edges = event_gnn.track_edges.cpu().numpy()

    track_mask = find_edge_indices(edge_index, track_edges)
    all_track_scores.append(after_gnn_scores[track_mask])

    # Store track edge truth values
    if all_target_truth_track[num_event] is None:
        track_edge_y = event_gnn.edge_y.cpu().numpy()[track_mask]
        track_edge_weights = event_gnn.edge_weights.cpu().numpy()[track_mask]
        all_target_truth_track[num_event] = track_edge_y & (track_edge_weights==1)
        all_non_target_truth_track[num_event] = track_edge_y & (track_edge_weights==0)
        
    if target_pt[num_event] is None:
        target_pt[num_event] = event_gnn.track_particle_pt.cpu().numpy()
        target_eta[num_event] = event_gnn.hit_eta[edge_index[0]].cpu().numpy()

# Concatenate all scores and truth values
all_scores = np.concatenate(all_scores)
all_target_truth = np.concatenate(all_target_truth)
all_non_target_truth = np.concatenate(all_non_target_truth)
all_false = np.concatenate(all_false)
all_track_scores = np.concatenate(all_track_scores)
all_target_truth_track = np.concatenate(all_target_truth_track)
all_non_target_truth_track = np.concatenate(all_non_target_truth_track)
target_pt = np.concatenate(target_pt)
target_eta = np.concatenate(target_eta)

# plot eta histogram of the true target edges with scores below 0.05

plt.figure(figsize=(10, 6))
plt.hist(target_eta[all_target_truth & (all_scores < 0.05)], bins=50, alpha=0.5, label='True Target Edges (Score < 0.05)', color='blue', histtype='step')
plt.savefig("eta_repartition_true_edges.svg")
print("Number of true target edges with score < 0.05:", np.sum(all_target_truth & (all_scores < 0.05)))