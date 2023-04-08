#!/bin/bash


python3.10 experiment_builder.py --experiment_name nm_funnel \
    --model squared_nm_gaussian_mixture \
    --data funnel \
    --comp 5 \
    --it 100 \
    --lr 0.0005645 \
    --momentum 0.7936 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape full \
    --covar_reg 4.4 \
    --batch_size 32 \
    --optimal_init funnel \
    --sparsity 0.284
