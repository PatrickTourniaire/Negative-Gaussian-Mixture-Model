#!/bin/bash


python3 experiment_builder.py --experiment_name nm_ring \
    --model squared_nm_gaussian_mixture \
    --data ring \
    --comp 2 \
    --it 300 \
    --lr 0.00037 \
    --momentum 0.5669 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape diag \
    --covar_reg 4.4 \
    --batch_size 64 \
    --optimal_init none \
    --sparsity 0.3685
