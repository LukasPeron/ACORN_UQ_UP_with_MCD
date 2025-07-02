#!/bin/bash

#SBATCH -A m2616
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=None
#SBATCH --cpus-per-task=64
#SBATCH --signal=SIGUSR1@120
#SBATCH --mail-user=lukas.peron@ens.psl.eu
#SBATCH --mail-type=ALL
#SBATCH --output=/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/jobs_logs/%j.out

srun python /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/uncertainty_on_track.py
