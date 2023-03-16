python3 experiment_builder.py --experiment_name nmgmm_sgdmom_ring_comp2_lr0.1 \
    --model squared_nm_gaussian_mixture \
    --data ring \
    --comp 2 \
    --it 25000 \
    --lr 0.1 \
    --validate_pdf 0 \
    --optimizer sgd_mom \
    --initialisation random \
    --covar_shape diag \
    --covar_reg 0