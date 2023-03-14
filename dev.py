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
import imageio

# Local imports
from src.models.mixtures.gaussian_mixture import GaussianMixture
from src.models.mixtures.squared_gaussian_mixture import SquaredGaussianMixture
from src.models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture

from src.utils.pickle_handler import *
from src.utils.early_stopping import EarlyStopping
from src.utils.initialisation_procedures import GMMInitalisation, check_random_state

"""
# TODO: Add a new module which does different initialisation techniques such as K-means
// # TODO: Add possibility to change the optimialisation techniques - for when experimentation is approaching. 
"""

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name", help="Name of experiment")
parser.add_argument("--model", help="Name of the model to experiment")
parser.add_argument("--data", help="Name of the pickle data file to train on")
parser.add_argument("--comp", help="Number of mixture components")
parser.add_argument("--it", help="Number of iterations for training")
parser.add_argument("--lr", help="Learning rate for the SGD optimiser")
parser.add_argument("--validate_pdf", help="To use grid sampling to validate the pdf")
parser.add_argument("--optimizer", help="Which optimizer to use")
parser.add_argument("--initialisation", help="Initialisation technique to use")
parser.add_argument("--covar_shape", help="Shape of the initialised covariance matrix")
parser.add_argument("--covar_reg", help="Regularisation constant added to the digonal matrix")

args = parser.parse_args()

# Experiment configuration
model_config = {
    'model_name': args.model,
    'dataset': args.data,
    'components': int(args.comp),
    'iterations': int(args.it),
    'learning_rate': float(args.lr),
    'validate_pdf': bool(int(args.validate_pdf)),
    'optimizer': args.optimizer,
    'initialisation': args.initialisation,
    'covar_shape': args.covar_shape,
    'covar_reg': float(args.covar_reg)
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

console = Console()

with  console.status("Loading dataset...") as status:
    # Load data - which is generated with `generate.py`
    train_set = load_object('data/train', model_config['dataset'])
    val_set = load_object('data/val', model_config['dataset'])
    test_set = load_object('data/test', model_config['dataset'])

    console.log(f"Dataset \"{model_config['dataset']}\" loaded")

    # ===========================================================================
    #                   TRAIN AND MONITOR WITH TENSORBOARD
    # ===========================================================================

    status.update(status=f'Loading "{model_config["model_name"]}" model...')

    path_sequences = f'out/sequences/{args.experiment_name}'
    if not os.path.isdir(os.path.abspath(path_sequences)):
        os.makedirs(os.path.abspath(path_sequences))
    
    path_models = f'out/models/{args.experiment_name}'
    if not os.path.isdir(os.path.abspath(path_models)):
        os.makedirs(os.path.abspath(path_models))
    
    #=============================== INIT PARAMS ===============================

    initaliser = GMMInitalisation(
        n_components=model_config['components'],
        init_params=model_config['initialisation'],
        covariance_type=model_config['covar_shape'],
        reg_covar=model_config['covar_reg']
    )
    random_seed = check_random_state(None)
    initaliser.initialize_parameters(train_set, random_seed)

    _covariances = initaliser.covariances_
    _means = initaliser.means_
    
    if model_config['covar_shape'] == 'diag':
        _covariances = np.array([np.diag(S) for S in _covariances])


    #=============================== NMGMM SETUP ===============================
    
    # Model and optimiser
    model = available_models[model_config['model_name']](
        n_clusters = model_config['components'], 
        n_dims = 2,
        init_means=torch.from_numpy(_means),
        init_sigmas=torch.from_numpy(_covariances))

    model.to(device)
    model.set_monitoring(os.path.abspath('runs'), args.experiment_name)

    optimizer_algo = available_optimizers[model_config["optimizer"]]
    optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'])
    
    if (model_config["optimizer"] == 'sgd_mom'):
        optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'], momentum=0.9)

    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

    console.log(f'Model "{model_config["model_name"]}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))


    #============================== SKLEARN GMM ================================
    
    # Base model from sklearn with same number of components
    base_model = SKGaussianMixture(
        n_components=model_config['components'], 
        random_state=random_seed,
        means_init=_means,
        precisions_init=[inv(S) for S in _covariances]).fit(train_set)
    base_loss = - (logsumexp(base_model.score_samples(train_set)) / train_set.shape[0])
    model.set_base_loss(base_loss)

    console.log(f'Model "{BASE_MODEL_NAME}" loaded')
    

    #============================= TRAINING NMGMM ==============================
    
    status.update(status=f'Training "{model_config["model_name"]}" model...')

    for it in range(model_config['iterations']):
        model.add_base_means(base_model.means_, it)
        model.add_base_weights(base_model.weights_, it)

        optimizer.zero_grad()
        it_train_loss = model(torch.from_numpy(train_set), it, model_config['validate_pdf'])
      
        with torch.no_grad(): 
            it_val_loss = model.val_loss(torch.from_numpy(val_set), it)
        
        if it % 100 == 0:
            model.sequence_visualisation(
                train_set,
                val_set,
                os.path.abspath(f'{path_sequences}/seq_it{it}.png')
            )
        
        early_stopping(it_train_loss, it_val_loss)
        if early_stopping.early_stop: break

        it_train_loss.backward()
        optimizer.step()

    console.log(f'Model "{model_config["model_name"]}" was trained successfully')
    model.clear_monitoring()

    torch.save(model.state_dict(), f'out/saved_models/{args.experiment_name}')


    # ===========================================================================
    #                             VISUALISE MODEL
    # ===========================================================================

    status.update(status=f'Visualising "{model_config["model_name"]}" model...')
    model_name_path = model_config["model_name"]

    model.plot_heatmap(
        train_set,
        val_set,
        os.path.abspath(f'{path_models}/{args.experiment_name}_heatmap.pdf')
    )

    model.plot_contours(
        train_set,
        os.path.abspath(f'{path_models}/{args.experiment_name}_contours.pdf')
    )

    paths_frames = glob.glob(f'{path_sequences}/*.png', recursive=True)
    frames = [imageio.imread(f) for f in paths_frames]
    imageio.mimsave(
        f'{path_models}/{args.experiment_name}_anim.gif', 
        frames, 
        fps=(len(paths_frames) / 10.0)
    )

console.log(f'[bold green] Experiment ran successfully!')
