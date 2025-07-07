#!/bin/bash

#SBATCH -A m3443
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=None
#SBATCH --cpus-per-task=64
#SBATCH --signal=SIGUSR1@120
#SBATCH --requeue
#SBATCH --mail-user=lukas.peron@ens.psl.eu
#SBATCH --mail-type=ALL
#SBATCH --output=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/examples/Example_TrackML_UQ/jobs_logs/%j.out

path_to_acorn=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/examples/Example_TrackML_UQ
path_to_metric_learning_plots=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/data_dir/Example_TrackML_UQ/metric_learning
path_to_filter_plots=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/data_dir/Example_TrackML_UQ/filter
path_to_gnn_plots=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/data_dir/Example_TrackML_UQ/gnn
path_to_track_building_plots=/pscratch/sd/l/lperon/sanity_check_UQ_acorn/acorn/data_dir/Example_TrackML_UQ/track_building

srun acorn train $path_to_acorn/metric_learning_train.yaml
srun acorn infer $path_to_acorn/metric_learning_infer.yaml

mkdir -p $path_to_metric_learning_plots/plots
mkdir -p $path_to_metric_learning_plots/plots/uncalibrated/
mkdir -p $path_to_metric_learning_plots/plots/calibrated/
mkdir -p $path_to_metric_learning_plots/plots/uncalibrated/
mkdir -p $path_to_metric_learning_plots/plots/calibrated/

srun acorn eval $path_to_acorn/metric_learning_eval.yaml

srun acorn train $path_to_acorn/filter_train.yaml 
srun acorn infer $path_to_acorn/filter_infer.yaml

mkdir -p $path_to_filter_plots/plots
mkdir -p $path_to_filter_plots/plots/uncalibrated/
mkdir -p $path_to_filter_plots/plots/calibrated/
mkdir -p $path_to_filter_plots/plots/uncalibrated/
mkdir -p $path_to_filter_plots/plots/calibrated/

srun acorn eval $path_to_acorn/filter_eval.yaml

srun acorn train $path_to_acorn/gnn_train.yaml 
srun acorn infer $path_to_acorn/gnn_infer.yaml

mkdir -p $path_to_gnn_plots/plots
mkdir -p $path_to_gnn_plots/plots/uncalibrated/
mkdir -p $path_to_gnn_plots/plots/calibrated/
mkdir -p $path_to_gnn_plots/plots/uncalibrated/
mkdir -p $path_to_gnn_plots/plots/calibrated/

srun acorn eval $path_to_acorn/gnn_eval.yaml

srun acorn infer $path_to_acorn/track_building_infer.yaml

mkdir -p $path_to_track_building_plots/uncal_res/
srun acorn eval $path_to_acorn/track_building_eval.yaml

# if calibration is wanted

sed -i "s#calibration: false#calibration: true#g" $path_to_acorn/*.yaml
srun acorn calib $path_to_acorn/gnn_calibrate.yaml
srun acorn infer $path_to_acorn/gnn_infer.yaml
srun acorn eval $path_to_acorn/gnn_eval.yaml

srun acorn infer $path_to_acorn/track_building_infer.yaml
mkdir -p $path_to_track_building_plots/cal_res/
srun acorn eval $path_to_acorn/track_building_eval.yaml
