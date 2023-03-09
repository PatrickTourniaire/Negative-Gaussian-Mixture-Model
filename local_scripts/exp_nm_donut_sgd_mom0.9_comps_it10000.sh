#!/bin/bash

#if [ -z "$STY" ]; then exec screen -dm -S screenName /bin/bash "$0"; fi
#conda activate nmmm

python3 dev.py --experiment_name nm_mixture_sgd_comp2_it10000_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 5000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd_mom &

python3 dev.py --experiment_name nm_mixture_sgd_comp3_it10000_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 3 \
    --it 5000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd_mom &


python3 dev.py --experiment_name nm_mixture_sgd_comp4_it10000_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 4 \
    --it 5000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd_mom &


python3 dev.py --experiment_name nm_mixture_sgd_comp5_it10000_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 5 \
    --it 5000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd_mom &
