#!/bin/bash

#if [ -z "$STY" ]; then exec screen -dm -S screenName /bin/bash "$0"; fi
#conda activate nmmm

python3 dev.py --experiment_name nm_mixture_testing \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 3000 \
    --lr 0.01 \
    --validate_pdf 0 \
    --optimizer adam

