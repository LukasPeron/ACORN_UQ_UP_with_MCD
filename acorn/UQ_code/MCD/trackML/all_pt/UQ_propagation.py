from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.stats import kurtosis, skew
from acorn.utils.eval_utils import plot_uncertainty_vs_score, plot_edge_skewness_kurtosis, plot_uncertainty_distribution, plot_uncertainty_vs_pt, plot_uncertainty_vs_eta, plot_entropy_difference, generate_matching_gaussians, find_edge_indices, compare_entropy, plot_aleatoric_epistemic_uncertainty, plot_edge_scores_distribution, plot_number_edges_vs_eta, plot_edges_score_vs_eta, plot_edges_score_vs_pt
from acorn.stages.track_building.utils import rearrange_by_distance

n_train = 1400 # input()
output_dir = Path(f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/UQ_propagation/")

filter_dataset_name = "valset"  
gnn_dataset_name = "valset" 

# Load filter model
filter_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/filter/artifacts/"
if list(Path(filter_ckpt_path).glob("best*"))!=[]:   
    filter_checkpoint = list(Path(filter_ckpt_path).glob("best*"))[0]
    filter_model = Filter.load_from_checkpoint(filter_checkpoint)
    filter_model.hparams["data_split"] = [0,50,0]  # Set data split to [train, val, test] for filter model
    filter_model.setup(stage="test", input_dir="input_dir")
    dataset = getattr(filter_model, filter_dataset_name)
    print(f"Filter model loaded from {filter_checkpoint}")

# Load GNN model
gnn_ckpt_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/{n_train}/gnn/artifacts/"
if list(Path(gnn_ckpt_path).glob("best*"))!=[]:
    gnn_checkpoint = list(Path(gnn_ckpt_path).glob("best*"))[0]
    gnn_model = InteractionGNN.load_from_checkpoint(gnn_checkpoint)
    gnn_model.hparams["data_split"] = [0,50,0]
    gnn_model.setup(stage="predict")
    gnn_model.hparams["input_cut"] = 0.05 #! Need to set input_cut to 0.0 for UQ propagation in order to avoid varying graph input sizes to the GNN
    gnn_dataset = getattr(gnn_model, gnn_dataset_name)
    print(gnn_dataset.hparams["input_cut"])
    print(f"GNN model loaded from {gnn_checkpoint}")

# Set config & plot_config option for later plots functions
config = {
    "stage_dir": output_dir,
    "dataset": "valset",
    "score_cut": 0.5,  # Example score cut
    "input_cut": 0.0,  # Example input cut
    "n_train": n_train,
    "target_tracks": {
        'track_particle_pt': [1e3],  # Example pT threshold in MeV
        'track_particle_eta': [4.0]   # Example eta threshold
    },
}

plot_config = {
    "filename": "mc_dropout",
}
dropout_str = "0p1"
dropout_value = 0.1
calibration = False
UQ_propagation = True


# We use the following weights for the filter model, cf filter config (ie .yaml) files
filter_true_target_weight = 2
filter_true_non_target_weight = 0
filter_false_weight = 1

n_mcd_passes = 100 # Number of MCD passes
n_event = len(dataset)

all_edges_dict = {}
track_edges_dict = {}     # Store pt information for each edge  

for t in tqdm(range(n_mcd_passes)):
    for num_event, event in enumerate(dataset):
        with torch.inference_mode():
            filter_model.train()   # keep the filter model in training mode to apply dropout
            gnn_model.eval()       # keep the GNN model in evaluation mode

            # Apply the filter model stochastically to the event
            eval_dict_filter = filter_model.shared_evaluation(event.to(filter_model.device), 0)
            event_filter = eval_dict_filter["batch"]
            # print("number of edges after filter: ", event_filter.edge_index.shape[1])
            # Apply the gnn model deterministically to the filtered event
            event_filter = gnn_dataset.handle_edge_list(event_filter.cpu())
            gnn_dict = gnn_model.shared_evaluation(event_filter.to(gnn_model.device), 0)
            event_gnn = gnn_dict["batch"]

            # Get edge indices and scores from GNN output
            edge_index = event_gnn.edge_index.cpu().numpy()
            edge_scores = event_gnn.edge_scores.cpu().numpy()
            edge_y = event_gnn.edge_y.cpu().numpy()
            edge_weights = event_gnn.edge_weights.cpu().numpy()
            hit_eta = event_gnn.hit_eta.cpu().numpy()
            track_particle_pt = event_gnn.track_particle_pt.cpu().numpy()
            # Get track edges information
            track_edges = event_gnn.track_edges.cpu().numpy()
            # track_mask = find_edge_indices(edge_index, track_edges)
            track_edge_map = {}
            for idx in range(track_edges.shape[1]):
                edge_pair = (track_edges[0, idx], track_edges[1, idx])
                track_edge_map[edge_pair] = idx

            # loop on the edges to store the scores and track information
            for i in range(edge_index.shape[1]):
                edge_key = (num_event, edge_index[0, i].item(), edge_index[1, i].item())
                if edge_key not in all_edges_dict:
                    all_edges_dict[edge_key] = {"scores": [], "entropy": [], "y": edge_y[i], "weights": edge_weights[i], "eta": hit_eta[edge_index[0, i]].item()}
                all_edges_dict[edge_key]["scores"].append(edge_scores[i])
                all_edges_dict[edge_key]["entropy"].append(-edge_scores[i] * np.log(edge_scores[i] + 1e-10) - (1 - edge_scores[i]) * np.log(1 - edge_scores[i] + 1e-10))
                if edge_y[i]:
                    edge_pair = (edge_index[0, i].item(), edge_index[1, i].item())
                    if edge_pair in track_edge_map:
                        track_idx = track_edge_map[edge_pair]
                        if edge_key not in track_edges_dict:
                            track_edges_dict[edge_key] = {
                                "scores": [], 
                                "weights": edge_weights[i],
                                "pt": track_particle_pt[track_idx].item()
                            }
                        track_edges_dict[edge_key]["scores"].append(edge_scores[i])

all_mean_scores = {"mean_scores": [], "mean_uncertainties": [], "mean_score_entropy": [], "kurtosis": [], "skewness": [], "y": [], "weights": [], "eta": []}
all_mean_scores_track = {"mean_scores": [], "mean_uncertainties": [], "weights": [], "pt": []}

for edge_key, edge_info in all_edges_dict.items():
    mean_score = np.mean(edge_info["scores"])
    uncertainty = np.std(edge_info["scores"])
    score_entropy = np.mean(edge_info["entropy"])
    kurtosis_value = kurtosis(edge_info["scores"])
    skewness_value = skew(edge_info["scores"])

    all_mean_scores["mean_scores"].append(mean_score)
    all_mean_scores["mean_uncertainties"].append(uncertainty)
    all_mean_scores["mean_score_entropy"].append(score_entropy)
    all_mean_scores["kurtosis"].append(kurtosis_value)
    all_mean_scores["skewness"].append(skewness_value)
    all_mean_scores["y"].append(edge_info["y"])
    all_mean_scores["weights"].append(edge_info["weights"])
    all_mean_scores["eta"].append(edge_info["eta"])

for edge_key, edge_info in track_edges_dict.items():
    mean_score = np.mean(edge_info["scores"])
    uncertainty = np.std(edge_info["scores"])

    all_mean_scores_track["mean_scores"].append(mean_score)
    all_mean_scores_track["mean_uncertainties"].append(uncertainty)
    all_mean_scores_track["weights"].append(edge_info["weights"])
    all_mean_scores_track["pt"].append(edge_info["pt"])

# Convert lists to numpy arrays for easier handling
all_mean_scores["mean_scores"] = np.array(all_mean_scores["mean_scores"])
all_mean_scores["mean_uncertainties"] = np.array(all_mean_scores["mean_uncertainties"])
all_mean_scores["mean_score_entropy"] = np.array(all_mean_scores["mean_score_entropy"])
all_mean_scores["kurtosis"] = np.array(all_mean_scores["kurtosis"])
all_mean_scores["skewness"] = np.array(all_mean_scores["skewness"])
all_mean_scores["y"] = np.array(all_mean_scores["y"])
all_mean_scores["weights"] = np.array(all_mean_scores["weights"])
all_mean_scores["eta"] = np.array(all_mean_scores["eta"])
all_mean_scores_track["mean_scores"] = np.array(all_mean_scores_track["mean_scores"])
all_mean_scores_track["mean_uncertainties"] = np.array(all_mean_scores_track["mean_uncertainties"])
all_mean_scores_track["weights"] = np.array(all_mean_scores_track["weights"])
all_mean_scores_track["pt"] = np.array(all_mean_scores_track["pt"])

# Flatten arrays for analysis
all_flat_scores = all_mean_scores["mean_scores"]
all_flat_uncertainties = all_mean_scores["mean_uncertainties"]
all_flat_BCE_score_entropy = all_mean_scores["mean_score_entropy"]
all_flat_kurtosis = all_mean_scores["kurtosis"]
all_flat_skewness = all_mean_scores["skewness"]
all_flat_target_truth = all_mean_scores["y"] & (all_mean_scores["weights"] == filter_true_target_weight)  # Select only true target edges
all_flat_non_target_truth = all_mean_scores["y"] & (all_mean_scores["weights"] == filter_true_non_target_weight)  # Select only true non-target edges
all_flat_false = (~all_mean_scores["y"]) & (all_mean_scores["weights"] == filter_false_weight)  # Select only false edges
all_flat_eta = all_mean_scores["eta"]

all_flat_scores_track = all_mean_scores_track["mean_scores"]
all_flat_uncertainties_track = all_mean_scores_track["mean_uncertainties"]
print(all_flat_uncertainties_track)
all_flat_target_truth_track = all_mean_scores_track["weights"] == filter_true_target_weight  # Select only true target edges in track edges
all_flat_non_target_truth_track = all_mean_scores_track["weights"] == filter_true_non_target_weight  # Select only true non-target edges in track edges
all_flat_pt = all_mean_scores_track["pt"]

# Compute total uncertainty and epistemic uncertainty
all_flat_total_uncertainty = -all_mean_scores["mean_scores"] * np.log(all_mean_scores["mean_scores"] + 1e-10) - (1 - all_mean_scores["mean_scores"]) * np.log(1 - all_mean_scores["mean_scores"] + 1e-10)
all_flat_epistemic_uncertainty = all_flat_total_uncertainty - all_mean_scores["mean_score_entropy"]  # mutual information

# Save the results as .txt files
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_scores_{n_train}.txt"), all_flat_scores)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_uncertainties_{n_train}.txt"), all_flat_uncertainties)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_target_truth_{n_train}.txt"), all_flat_target_truth)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_non_target_truth_{n_train}.txt"), all_flat_non_target_truth)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_false_{n_train}.txt"), all_flat_false)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_pt_{n_train}.txt"), all_flat_pt)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_eta_{n_train}.txt"), all_flat_eta)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_scores_track_{n_train}.txt"), all_flat_scores_track)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_uncertainties_track_{n_train}.txt"), all_flat_uncertainties_track)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_BCE_score_entropy_{n_train}.txt"), all_flat_BCE_score_entropy)
np.savetxt(os.path.join(config["stage_dir"], f"Total_uncertainty_{n_train}.txt"), all_flat_total_uncertainty)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_epistemic_uncertainty_{n_train}.txt"), all_flat_epistemic_uncertainty)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_target_truth_track_{n_train}.txt"), all_flat_target_truth_track)
np.savetxt(os.path.join(config["stage_dir"], f"all_flat_non_target_truth_track_{n_train}.txt"), all_flat_non_target_truth_track)

# Plotting the results
plot_uncertainty_vs_score(all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

# Plot the results
plot_uncertainty_vs_pt(all_flat_pt, all_flat_scores_track, all_flat_uncertainties_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_uncertainty_vs_eta(all_flat_eta, all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_uncertainty_distribution(all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_edge_skewness_kurtosis(all_flat_scores, all_flat_skewness, all_flat_kurtosis, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_aleatoric_epistemic_uncertainty(all_flat_scores, all_flat_epistemic_uncertainty, all_flat_BCE_score_entropy, all_flat_total_uncertainty, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_edge_scores_distribution(all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, calibration, UQ_propagation)

plot_number_edges_vs_eta(all_flat_eta, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, calibration, UQ_propagation)

plot_edges_score_vs_eta(all_flat_eta, all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)

plot_edges_score_vs_pt(all_flat_pt, all_flat_scores_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, calibration, UQ_propagation)
