#!/bin/bash

#if [ -z "$STY" ]; then exec screen -dm -S screenName /bin/bash "$0"; fi
#conda activate nmmm

python3 dev.py --experiment_name nm_mixture_testing \
    --model gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 10000 \
    --lr 0.001 \
    --validate_pdf 1 \
    --optimizer sgd

