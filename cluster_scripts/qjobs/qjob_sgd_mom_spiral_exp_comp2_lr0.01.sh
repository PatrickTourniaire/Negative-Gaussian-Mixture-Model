
#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=04:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU: 
#$ -pe gpu 2
#
# Request 16 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=32G

# Initialise the environment modules and load CUDA version 8.0.61
. /etc/profile.d/modules.sh
module load anaconda/5.0.1
module load cuda/11.0.2

source activate nmmm

# Run the program
./cluster_scripts/experiments/sgd_mom/spiral/exp_comp2_lr0.01.sh
