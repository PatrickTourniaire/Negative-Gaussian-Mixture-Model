#!/bin/bash


python3 experiment_builder.py --experiment_name nm_cosine \
    --model squared_nm_gaussian_mixture \
    --data cosine \
    --comp 6 \
    --it 50 \
    --lr 0.005 \
    --momentum 0.9 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape full \
    --covar_reg 4.4 \
    --batch_size 32 \
    --optimal_init cosine \
    --sparsity 0.1