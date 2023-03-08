#!/bin/bash

#if [ -z "$STY" ]; then exec screen -dm -S screenName /bin/bash "$0"; fi
#conda activate nmmm

python3 dev.py --experiment_name nm_mixture_sgd_comp2_it10000_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 10000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd &

python3 dev.py --experiment_name nm_mixture_sgd_comp2_it10000_lr0.01 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 10000 \
    --lr 0.01 \
    --validate_pdf 0 \
    --optimizer sgd &


python3 dev.py --experiment_name nm_mixture_sgd_comp2_it10000_lr0.001 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 10000 \
    --lr 0.001 \
    --validate_pdf 0 \
    --optimizer sgd &


python3 dev.py --experiment_name nm_mixture_sgd_comp2_it10000_lr0.0001 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 10000 \
    --lr 0.0001 \
    --validate_pdf 0 \
    --optimizer sgd &
