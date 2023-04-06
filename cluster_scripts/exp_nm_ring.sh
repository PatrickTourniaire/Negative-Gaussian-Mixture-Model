#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=10:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU:
#$ -pe gpu 1
#
# Request 16 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=16G

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
module load anaconda/5.0.1
module load cuda/11.0.2

source activate nmmm

python3.10 experiment_builder.py --experiment_name nm_ring \
    --model squared_nm_gaussian_mixture \
    --data ring \
    --comp 3 \
    --it 500 \
    --lr 0.0005 \
    --momentum 0.79 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape diag \
    --covar_reg 4.4 \
    --batch_size 8 \
    --optimal_init none \
    --sparsity 0.19