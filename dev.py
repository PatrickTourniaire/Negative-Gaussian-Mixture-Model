# External imports
import torch
import os
import numpy as np
import json
from rich.console import Console
from rich.layout import Layout
from sklearn.mixture import GaussianMixture as SKGaussianMixture
import argparse, sys
from scipy.special import logsumexp

# Local imports
from models.mixtures.gaussian_mixture import GaussianMixture
from models.mixtures.squared_gaussian_mixture import SquaredGaussianMixture
from models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture



from utils.pickle_handler import *

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Name of the model to experiment")
parser.add_argument("--data", help="Name of the pickle data file to train on")
parser.add_argument("--comp", help="Number of mixture components")
parser.add_argument("--it", help="Number of iterations for training")
parser.add_argument("--lr", help="Learning rate for the SGD optimiser")
parser.add_argument("--validate_pdf", help="To use grid sampling to validate the pdf")

args = parser.parse_args()

# Experiment configuration
model_config = {
    'model_name': args.model,
    'dataset': args.data,
    'components': int(args.comp),
    'iterations': int(args.it),
    'learning_rate': float(args.lr),
    'validate_pdf': bool(int(args.validate_pdf))
}

available_models = {
    'gaussian_mixture': GaussianMixture,
    'squared_gaussian_mixture': SquaredGaussianMixture,
    'squared_nm_gaussian_mixture': NMSquaredGaussianMixture
}

BASE_MODEL_NAME = 'sklearn_gmm'

console = Console()

with  console.status("Loading dataset...") as status:
    # Load data - which is generated with `generate.py`
    features = load_object('data', model_config['dataset'])

    console.log(f"Dataset \"{model_config['dataset']}\" loaded")

    # ===========================================================================
    #                   TRAIN AND MONITOR WITH TENSORBOARD
    # ===========================================================================

    status.update(status=f'Loading "{model_config["model_name"]}" model...')

    # Model and optimiser
    model = available_models[model_config['model_name']](model_config['components'], 2)
    model.set_monitoring(os.path.abspath('runs'), model_config["model_name"])
    model.set_vis_config(res=200, vmin=-4, vmax=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    console.log(f'Model "{model_config["model_name"]}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))

    # Base model from sklearn with same number of components
    base_model = SKGaussianMixture(n_components=model_config['components'], random_state=0, means_init=[[0,0], [0,0]]).fit(features)
    base_loss = - (logsumexp(base_model.score_samples(features)) / features.shape[0])
    model.set_base_loss(base_loss)

    console.log(f'Model "{BASE_MODEL_NAME}" loaded')

    status.update(status=f'Training "{model_config["model_name"]}" model...')

    for it in range(model_config['iterations']):
        model.add_base_means(base_model.means_, it)
        model.add_base_weights(base_model.weights_, it)
        optimizer.zero_grad()
        loss = model(torch.from_numpy(features), it, model_config['validate_pdf'])
        loss.backward()
        optimizer.step()

    console.log(f'Center log likelihood: {str(model.pdf(torch.Tensor([[0, 0]])))}')
    console.log(f'Donut log likelihood: {str(model.pdf(torch.Tensor([[3, 0]])))}')

    console.log(f'Model "{model_config["model_name"]}" was trained successfully')
    model.clear_monitoring()

    # ===========================================================================
    #                     VISUALISE NON-MONOTONIC MODEL
    # ===========================================================================

    status.update(status=f'Visualising "{model_config["model_name"]}" model...')

    grid, _, _ = model.create_grid()
    log_likelihoods = model.log_likelihoods(grid)

    model_name_path = model_config["model_name"]

    model.plot_heatmap(
        features,
        os.path.abspath(f'out/models/{model_config["model_name"]}_heatmap.pdf')
    )

    model.plot_contours(
        features,
        os.path.abspath(f'out/models/{model_config["model_name"]}_contours.pdf')
    )

console.log(f'[bold green] Experiment ran successfully!')
