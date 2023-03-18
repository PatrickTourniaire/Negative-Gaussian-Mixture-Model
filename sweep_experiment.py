# External imports
import torch
import os
import numpy as np
from numpy.linalg import inv
import json
from rich.console import Console
from sklearn.mixture import GaussianMixture as SKGaussianMixture
import argparse
from scipy.special import logsumexp
import glob
import wandb
import yaml

# Local imports
from src.models.mixtures.gaussian_mixture import GaussianMixture
from src.models.mixtures.squared_gaussian_mixture import SquaredGaussianMixture
from src.models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture

from src.utils.pickle_handler import *
from src.utils.early_stopping import EarlyStopping
from src.utils.initialisation_procedures import GMMInitalisation, check_random_state



with open('sweeps/sweep_ring.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)

# Experiment configuration
model_config = {
    'model_name': wandb.config.model,
    'dataset': wandb.config.data,
    'components': wandb.config.components,
    'iterations': wandb.config.iterations,
    'learning_rate': wandb.config.learning_rate,
    'optimizer': wandb.config.optimizer,
    'initialisation': wandb.config.initialisation,
    'covar_shape': wandb.config.covar_shape,
    'covar_reg': wandb.config.covar_reg,
    'optimal_init': wandb.config.optimal_init,
    'validate_pdf': False
}

available_models = {
    'gaussian_mixture': GaussianMixture,
    'squared_gaussian_mixture': SquaredGaussianMixture,
    'squared_nm_gaussian_mixture': NMSquaredGaussianMixture
}

available_optimizers = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'sgd_mom': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad
}

BASE_MODEL_NAME = 'sklearn_gmm'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_REPO = str(os.path.abspath('out/'))

console = Console()


with  console.status("Loading dataset...") as status:
    # Load data - which is generated with `generate.py`
    train_set = load_object('data/train', model_config['dataset'])
    val_set = load_object('data/val', model_config['dataset'])
    test_set = load_object('data/test', model_config['dataset'])

    tensor_train_set = torch.from_numpy(train_set) if str(device) == 'cpu' else torch.from_numpy(train_set).cuda()
    tensor_val_set = torch.from_numpy(val_set) if str(device) == 'cpu' else torch.from_numpy(val_set).cuda()
    tensor_test_set = torch.from_numpy(test_set) if str(device) == 'cpu' else torch.from_numpy(test_set).cuda()

    console.log(f"Dataset \"{model_config['dataset']}\" loaded")

    # ===========================================================================
    #                   TRAIN AND MONITOR WITH TENSORBOARD
    # ===========================================================================

    status.update(status=f'Loading "{model_config["model_name"]}" model...')
    
    #=============================== INIT PARAMS ===============================

    initaliser_nmgmm = GMMInitalisation(
        n_components=model_config['components'],
        init_params=model_config['initialisation'],
        covariance_type=model_config['covar_shape'],
        reg_covar=model_config['covar_reg']
    )
    initaliser_gmm = GMMInitalisation(
        n_components=model_config['components'] ** 2,
        init_params=model_config['initialisation'],
        covariance_type=model_config['covar_shape'],
        reg_covar=model_config['covar_reg']
    )
    random_seed = check_random_state(None)
    
    initaliser_nmgmm.initialize_parameters(train_set, random_seed)
    initaliser_gmm.initialize_parameters(train_set, random_seed)

    _covariances_nmgmm = initaliser_nmgmm.covariances_
    _covariances_gmm = initaliser_gmm.covariances_
    
    _means_nmgmm = initaliser_nmgmm.means_
    _means_gmm = initaliser_gmm.means_
    
    if model_config['covar_shape'] == 'diag':
        _covariances_nmgmm = np.array([np.diag(np.sqrt(S)) for S in _covariances_nmgmm])
        _covariances_gmm = np.array([np.diag(S) for S in _covariances_gmm])
    
    if model_config['optimal_init'] == 'funnel':
        _covariances_nmgmm = _covariances_nmgmm
        _means_nmgmm[0] = [5, -5]
        _means_nmgmm[1] = [5, 5]
        _weights_nmgmm = torch.zeros(model_config['components'], dtype=torch.float64).normal_(1, 0.5)
        _weights_nmgmm[0] = torch.Tensor([-1])
        _weights_nmgmm[1] = torch.Tensor([-1])


    #=============================== NMGMM SETUP ===============================
    
    # Model and optimiser
    model = available_models[model_config['model_name']](
        n_clusters = model_config['components'], 
        n_dims = 2,
        init_means=torch.from_numpy(_means_nmgmm),
        init_sigmas=torch.from_numpy(_covariances_nmgmm))

    model.to(device)
    model.set_monitoring(os.path.abspath('runs'), 'test')

    optimizer_algo = available_optimizers[model_config["optimizer"]]
    optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'])
    
    if (model_config["optimizer"] == 'sgd_mom'):
        optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'], momentum=wandb.config.momentum)

    console.log(f'Model "{model_config["model_name"]}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))


    #============================== SKLEARN GMM ================================
    
    # Base model from sklearn with same number of components
    base_model = SKGaussianMixture(
        n_components=model_config['components'] ** 2, 
        random_state=random_seed,
        means_init=_means_gmm,
        precisions_init=[inv(S) for S in _covariances_gmm]).fit(train_set)
    base_loss = - (logsumexp(base_model.score_samples(train_set)) / train_set.shape[0])
    model.set_base_loss(base_loss)

    console.log(f'Model "{BASE_MODEL_NAME}" loaded')
    

    #============================= TRAINING NMGMM ==============================
    
    status.update(status=f'Training "{model_config["model_name"]}" model...')

    for it in range(model_config['iterations']):
        model.add_base_means(base_model.means_, it)
        model.add_base_weights(base_model.weights_, it)

        optimizer.zero_grad()
        it_train_loss = model(tensor_train_set, it, model_config['validate_pdf'])
      
        with torch.no_grad(): 
            it_val_loss = model.val_loss(tensor_val_set, it)
        
        model.log_means(wandb, it)
        model.log_weights(wandb, it)
        wandb.log({
            "train": {
               "loss":  it_train_loss
            },
            "validation": {
                "loss": it_val_loss
            },
            "baseline": {
                "loss": base_loss
            },
            "iteration": it
        })

        if it % 10 == 0:
            fig = model.sequence_visualisation(
                tensor_train_set,
                tensor_val_set,
            )
            wandb.log({f'sequence_plot': wandb.Image(fig)})

        it_train_loss.backward()
        optimizer.step()

    console.log(f'Model "{model_config["model_name"]}" was trained successfully')
    model.clear_monitoring()


wandb.finish()
console.log(f'[bold green] Experiment ran successfully!')
