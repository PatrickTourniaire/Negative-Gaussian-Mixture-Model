#!/bin/bash


python3 experiment_builder.py --experiment_name nm_ring \
    --model squared_nm_gaussian_mixture \
    --data ring \
    --comp 4 \
    --it 200 \
    --lr 0.0006439775513930731 \
    --momentum 0.916475640113362 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape diag \
    --covar_reg 4.4 \
    --batch_size 64 \
    --optimal_init none \
    --sparsity 0.15174970071675364
