# Non Monotonic Mixture Models

BSc Honours project on achieving generalisable NMMMs by squaring the probabilistic circuit (PC) of mixtures with negative weights

## Description

This is a preliminary README so everything will not be discussed here, however please contact me about the project write-up for more information about the project.

## Getting Started

### Dependencies

* The project depends on Weights and Biases (WandB) for logging. This package is included in the requirements, however it is required that you login with your own account.

### Setup

1. Setup conda environemt with required packages
```
conda create --name <env> --file requirements.txt
```
2. Login to your WandB account
```
wandb login
```
3. In `experiment_builder.py` update line 87-91 with your WandB project and entity/username.
```
wandb.init(
    project=<project_name>,
    entity=<username/entity>,
    config={**model_config}
)
```

### Executing program

* Run a local experiment using one of the dataset shapes provided under `data/`. See `local_scripts/` to see how to run an experiment with any of these shapes.
```
chmod +x local_script/<experiment_name>.sh && ./local_script/<experiment_name>.sh
```
* For running experiments on Eddie, some sample scripts are given under `cluster_scripts/`
* To run WandB sweeps for hyperparameter optimisation, you can setup a sweep script under `sweeps/` and then setup a start script for that sweep under `cluster_scripts/` to run it on Eddie.

Note: for running things on eddie you need to update line 100 in `experiment_builder.py` to output to an appropriate location with enough space for model checkpoints.

## Help

If you encounter any issues please feel free to reach out on Zulip or by email (s1900878@ed.ac.uk or patrick@tourniaire.net)

## Authors

* Patrick Tourniaire | s1900878@ed.ac.uk | patrick@tourniaire.net 

## Acknowledgments

Inspiration, code snippets, etc.
* [PyKeops - GMM with SGD](https://www.kernel-operations.io/keops/_auto_tutorials/gaussian_mixture/plot_gaussian_mixture.html)
* [Dataset shapes](https://arxiv.org/abs/1811.08357)
* [Batched mahalanobis distance](https://github.com/pytorch/pytorch/blob/main/torch/distributions/multivariate_normal.py)
