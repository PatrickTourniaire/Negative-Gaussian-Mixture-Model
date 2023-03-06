#!/bin/bash

python3 dev.py --experiment_name nm_mixture_sgd_comp2_it100_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data donut \
    --comp 2 \
    --it 1 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd