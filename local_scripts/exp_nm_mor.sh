#!/bin/bash


python3 experiment_builder.py --experiment_name nm_mor \
    --model squared_nm_gaussian_mixture \
    --data mor \
    --comp 8 \
    --it 200 \
    --lr 0.0002 \
    --momentum 0.97 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape diag \
    --covar_reg 4.4 \
    --batch_size 32 \
    --optimal_init mor \
    --sparsity 0.53