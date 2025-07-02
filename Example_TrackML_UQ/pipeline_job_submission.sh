#!/bin/bash

#SBATCH -A m2616
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=None
#SBATCH --cpus-per-task=32
#SBATCH --signal=SIGUSR1@120
#SBATCH --mail-user=your@mail.com
#SBATCH --mail-type=ALL

path_to_acorn=/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/$1
path_to_metric_learning_plots=/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/$1/metric_learning
path_to_filter_plots=/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/$1/filter
path_to_gnn_plots=/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/$1/gnn

srun acorn train $path_to_acorn/metric_learning_train.yaml
# acorn infer $path_to_acorn/metric_learning_infer.yaml

# mkdir -p $path_to_metric_learning_plots/plots
# mkdir -p $path_to_metric_learning_plots/plots/uncalibrated/
# mkdir -p $path_to_metric_learning_plots/plots/calibrated/
# mkdir -p $path_to_metric_learning_plots/plots/uncalibrated/$1
# mkdir -p $path_to_metric_learning_plots/plots/calibrated/$1


# acorn train $path_to_acorn/filter_train.yaml 
# acorn infer $path_to_acorn/filter_infer.yaml

# mkdir -p $path_to_filter_plots/plots
# mkdir -p $path_to_filter_plots/plots/uncalibrated/
# mkdir -p $path_to_filter_plots/plots/calibrated/
# mkdir -p $path_to_filter_plots/plots/uncalibrated/$1
# mkdir -p $path_to_filter_plots/plots/calibrated/$1

# acorn train $path_to_acorn/gnn_train.yaml 
# acorn infer $path_to_acorn/gnn_infer.yaml

# mkdir -p $path_to_gnn_plots/plots
# mkdir -p $path_to_gnn_plots/plots/uncalibrated/
# mkdir -p $path_to_gnn_plots/plots/calibrated/
# mkdir -p $path_to_gnn_plots/plots/uncalibrated/$1
# mkdir -p $path_to_gnn_plots/plots/calibrated/$1

# acorn eval $path_to_acorn/gnn_eval.yaml

# acorn infer $path_to_acorn/track_building_infer.yaml
