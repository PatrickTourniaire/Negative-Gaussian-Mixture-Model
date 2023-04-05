#!/bin/bash


python3 experiment_builder.py --experiment_name nm_spiral \
    --model squared_nm_gaussian_mixture \
    --data spiral \
    --comp 4 \
    --it 20 \
    --lr 0.009 \
    --momentum 0.23 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape full \
    --covar_reg 4.4 \
    --batch_size 32 \
    --optimal_init spiral \
    --sparsity 0.6