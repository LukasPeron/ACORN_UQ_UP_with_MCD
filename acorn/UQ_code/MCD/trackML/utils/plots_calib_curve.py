import numpy as np
import torch
import os
from pathlib import Path
from acorn.utils.eval_utils import plot_calibration_curve, plot_reliability_diagram
import tqdm

"""
Procedure :
1. Load the valset and testset
2.a Load the trained model
2.b If calibration option is specified, load the calibration model
3. Get the predictions on the valset and testset
4. Plot the calibration curve and reliability diagram
"""

scores = []
truth = []


data_path = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/valset")
for event_id in tqdm.tqdm(range(21000, 22755+1)):
    if list(data_path.glob(f"*0000{event_id}*"))!=[]:   
        post_step = list(data_path.glob(f"*0000{event_id}*"))[0]
        data_post_step = torch.load(post_step, map_location="cpu")
        scores.append(data_post_step.scores.cpu().numpy())
        truth.append((data_post_step.y == 1.0).cpu().numpy())

# flatten the lists of scores
scores = np.concatenate(scores)
truth = np.concatenate(truth)
config = {
    "n_train": 1400,
    "target_tracks": {"track_particle_pt": [1000]},
    "dataset_size": 50,
    "stage_dir": "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn",
    "dataset": "valset"
}

plot_calibration_curve(all_scores=scores, all_truth=truth, dataset="valset", config=config, plot_config={}, dropout_str="", dropout_value=0, from_calibration_stage=True)

plot_reliability_diagram(all_scores=scores, all_truth=truth, dataset="valset", config=config, plot_config={}, dropout_str="", dropout_value=0, from_calibration_stage=True)