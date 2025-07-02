from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from acorn.stages.track_building.models.cc_and_walk import CCandWalk
from acorn.stages.track_building.track_building_stage import GraphDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import numpy as np
import pandas as pd
import torch
import os
from acorn.stages.track_building.utils import rearrange_by_distance
from acorn.stages.track_building import utils

n_train = 1400 
data_split = [0, 1, 0]  # [train, val, test]

output_dir = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/track_building/")

filter_dataset_name = "valset"  
gnn_dataset_name = "valset" 

# Load config
config = {
    "data_split": data_split, # [train, val, test]
    "score_cut_cc": 0.01, # uncalibrated : 0.01
    "score_cut_walk": {
        "add": 0.6,
        "min": 0.1
    },
    "devices": 1,
    "log_level": "INFO",
    "max_workers": 1,
    "n_train": 1400,
    "UQ_propagation": True,
    "stage_dir": f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/track_building"
}

evaluation_config = {
    "matching_fraction": 0.5,
    "matching_style": "ATLAS",
    "min_track_length": 5,
    "min_particle_length": 5,
    "target_tracks": {
    "pt": [1000.0, np.inf],
    }
}

# Load filter model
filter_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/filter/artifacts/"
if list(Path(filter_ckpt_path).glob("best*"))!=[]:   
    filter_checkpoint = list(Path(filter_ckpt_path).glob("best*"))[0]
    filter_model = Filter.load_from_checkpoint(filter_checkpoint)
    filter_model.hparams["data_split"] = data_split  # Set data split to [train, val, test] for filter model
    filter_model.setup(stage="test", input_dir="input_dir")
    dataset = getattr(filter_model, filter_dataset_name)
    print(f"Filter model loaded from {filter_checkpoint}")

# Load GNN model
gnn_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/artifacts/"
if list(Path(gnn_ckpt_path).glob("best*"))!=[]:
    gnn_checkpoint = list(Path(gnn_ckpt_path).glob("best*"))[0]
    gnn_model = InteractionGNN.load_from_checkpoint(gnn_checkpoint)
    gnn_model.hparams["data_split"] = data_split
    gnn_model.setup(stage="predict")
    gnn_model.hparams["input_cut"] = 0.05 
    gnn_dataset = getattr(gnn_model, gnn_dataset_name)
    print(f"GNN model loaded from {gnn_checkpoint}")

# Load CC&Walk model
cc_and_walk_model = CCandWalk(config)
cc_and_walk_model.setup(stage="test")
cc_and_walk_dataset = getattr(cc_and_walk_model, "valset")

# Process only the first event
event = dataset[0]
print(f"Processing event: {event}")

with torch.inference_mode():
    filter_model.eval()   # Set filter model to evaluation mode (no dropout)
    gnn_model.eval()      # Set gnn model to evaluation mode (no dropout)
    
    # Apply the filter model deterministically to the event
    eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
    event_filter = eval_dict_filter["batch"]
    event_filter = gnn_dataset.handle_edge_list(event_filter.cpu()) #apply input cut
    gnn_dict = gnn_model.shared_evaluation(event_filter.to(gnn_model.device), 0)
    event_gnn = gnn_dict["batch"]
    track_graph = cc_and_walk_model._build_tracks_one_evt(event_gnn.cpu(), "")
    # After stochastic inference we get the efficiency and purity of the tracks
evaluated_events = []

# evaluated_events.append(
#     utils.evaluate_labelled_graph(
#         track_graph,
#         matching_fraction=evaluation_config.get("matching_fraction", 0.5),
#         matching_style=evaluation_config.get("matching_style", "ATLAS"),
#         sel_conf=evaluation_config.get("target_tracks", {}),
#         min_track_length=evaluation_config.get("min_track_length", 5),
#     )
# )

# evaluated_events = pd.concat(evaluated_events)
# particles = evaluated_events[evaluated_events["is_reconstructable"]]
# reconstructed_particles = particles[
#     particles["is_reconstructed"] & particles["is_matchable"]
# ]
# tracks = evaluated_events[evaluated_events["is_matchable"]]
# matched_tracks = tracks[tracks["is_matched"]]


tracks = utils.load_reconstruction_df(track_graph)

tracks = tracks[tracks.track_id != -1]  # Filter out tracks with track_id == -1
tracks = tracks.groupby("track_id")["hit_id"].apply(list)
print(tracks.values)
print(track_graph)