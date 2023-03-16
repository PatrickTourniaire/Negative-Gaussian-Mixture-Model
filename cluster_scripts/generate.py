import os
import glob

experiment_paths = glob.glob('experiments/*/*/*.sh', recursive=True)
experiment_names = []

for path in experiment_paths: 
    details = path.split('.sh')[0].split('/')[-1]
    optim = path.split('/')[1]
    data = path.split('/')[2]

    experiment_names.append(f'{optim}_{data}_{details}')

qscript = lambda experiment: f"""
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
./cluster_scripts/{experiment}
"""

for path, name in zip(experiment_paths, experiment_names):
    with open(f'qjobs/qjob_{name}.sh', 'w') as writer:
        writer.write(qscript(path))
