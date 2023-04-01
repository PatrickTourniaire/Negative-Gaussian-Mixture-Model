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
#$ -l h_vmem=18G

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
module load anaconda/5.0.1
module load cuda/11.0.2

source activate nmmm

# Run the program
wandb agent ptourniaire/NMMMs/3ho9gtk2
