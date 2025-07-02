import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
from acorn.utils.eval_utils import plot_uncertainty_vs_score, plot_aleatoric_epistemic_uncertainty

all_flat_scores = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_scores_1400.txt")
all_flat_uncertainties = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_uncertainties_1400.txt")
all_flat_target_truth = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_target_truth_1400.txt")
all_flat_non_target_truth = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_non_target_truth_1400.txt")
all_flat_false = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_false_1400.txt")
all_flat_epistemic_uncertainty = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_epistemic_uncertainty_1400.txt")
all_flat_BCE_score_entropy = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/all_flat_BCE_score_entropy_1400.txt")
all_flat_total_uncertainty = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut/Total_uncertainty_1400.txt")

all_flat_false = all_flat_false.astype(bool)
all_flat_target_truth = all_flat_target_truth.astype(bool)
all_flat_non_target_truth = all_flat_non_target_truth.astype(bool)

dataset="valset"
config = {"score_cut": 0.5,
          "n_train": 1400,
          "target_tracks": {"track_particle_pt": [1000, np.inf]},
          "nb_MCD_passes": 100,
          "dataset": "valset",
          "dataset_size": 50,
          "stage_dir": "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/plots/uncalibrated/with_input_cut"}
dropout_value=0.1
dropout_str = ""
plot_config = {"filename": "mc_dropout"
               }
plot_uncertainty_vs_score(
        all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False
    )

plot_aleatoric_epistemic_uncertainty(
        all_flat_scores, all_flat_epistemic_uncertainty, all_flat_BCE_score_entropy, 
        all_flat_total_uncertainty, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,
        dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False
    )