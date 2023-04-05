#!/bin/bash


python3 experiment_builder.py --experiment_name nm_banana \
    --model squared_nm_gaussian_mixture \
    --data banana \
    --comp 2 \
    --it 50 \
    --lr 0.0001 \
    --momentum 0.9 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape full \
    --covar_reg 4.4 \
    --batch_size 8 \
    --optimal_init banana \
    --sparsity 0.2