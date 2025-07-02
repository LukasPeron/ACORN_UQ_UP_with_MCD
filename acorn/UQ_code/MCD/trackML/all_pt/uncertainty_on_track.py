from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from acorn.stages.track_building.models.cc_and_walk import CCandWalk
from acorn.stages.track_building.track_building_stage import GraphDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from acorn.stages.track_building.utils import rearrange_by_distance
from acorn.stages.track_building import utils

n_train = 1400 
data_split = [0, 50, 0]  # [train, val, test]
n_mcd_passes = 100 # Number of MCD passes

output_dir = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/track_building/")

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
    gnn_model.hparams["input_cut"] = 0.05 #! Need to set input_cut to 0.0 for UQ propagation in order to avoid varying graph input sizes to the GNN
    gnn_dataset = getattr(gnn_model, gnn_dataset_name)
    print(f"GNN model loaded from {gnn_checkpoint}")

# Load CC&Walk model
cc_and_walk_model = CCandWalk(config)
cc_and_walk_model.setup(stage="test")
cc_and_walk_dataset = getattr(cc_and_walk_model, "valset")

n_event = len(dataset)
eff_list = []

for t in range(n_mcd_passes):
    stochastic_dataset = []
    for event in tqdm(dataset):
        with torch.inference_mode():
            filter_model.train()   # keep the filter model in training mode to apply dropout
            gnn_model.train()       # keep the gnn model in training mode to apply dropout
            # Apply the filter model stochastically to the event
            eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
            event_filter = eval_dict_filter["batch"]
            event_filter = gnn_dataset.handle_edge_list(event_filter.cpu()) #apply input cut
            gnn_dict = gnn_model.shared_evaluation(event_filter.to(gnn_model.device), 0)
            event_gnn = gnn_dict["batch"]
            track_graph = cc_and_walk_model._build_tracks_one_evt(event_gnn.cpu(), "")
            stochastic_dataset.append(track_graph)
    # After stochastic inference we get the efficiency and purity of the tracks
    evaluated_events = []

    for event in stochastic_dataset:
        evaluated_events.append(
            utils.evaluate_labelled_graph(
                event,
                matching_fraction=evaluation_config.get("matching_fraction", 0.5),
                matching_style=evaluation_config.get("matching_style", "ATLAS"),
                sel_conf=evaluation_config.get("target_tracks", {}),
                min_track_length=evaluation_config.get("min_track_length", 5),
            )
        )
    
    evaluated_events = pd.concat(evaluated_events)
    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[
        particles["is_reconstructed"] & particles["is_matchable"]
    ]
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=["event_id", "particle_id"]))
    n_reconstructed_particles = len(
        reconstructed_particles.drop_duplicates(subset=["event_id", "particle_id"])
    )
    eff = n_reconstructed_particles / n_particles
    eff_list.append(eff)
    print(f"Efficiency for pass {t}: {eff:.4f}")

eff_list = np.array(eff_list)
np.savetxt(output_dir / 'efficiency.txt', eff_list)
