# External imports
import torch
import os
import json
from rich.console import Console
from rich.layout import Layout
from sklearn.mixture import GaussianMixture
import argparse, sys

# Local imports
from models.mixtures.gm import MultivariateGaussianMixture
from utils.pickle_handler import *

parser = argparse.ArgumentParser()

parser.add_argument("--model", help="Name of the model to experiment")
parser.add_argument("--data", help="Name of the pickle data file to train on")
parser.add_argument("--comp", help="Number of mixture components")
parser.add_argument("--it", help="Number of iterations for training")
parser.add_argument("--lr", help="Learning rate for the SGD optimiser")

args = parser.parse_args()

# Config
MODEL_NAME = args.model
MODEL_BASE_NAME = 'BASE_MODEL_GMM'

model_config = {
    'model_name': args.model,
    'dataset': args.data,
    'components': int(args.comp),
    'iterations': int(args.it),
    'learning_rate': float(args.lr),
}

console = Console()

with console.status("Loading dataset...") as status:
    # Load data - which is generated with `generate.py`
    features = load_object('data', model_config['dataset'])

    console.log(f"Dataset \"{model_config['dataset']}\" loaded")

    # ===========================================================================
    #                   TRAIN AND MONITOR WITH TENSORBOARD
    # ===========================================================================

    status.update(status=f'Loading "{MODEL_NAME}" model...')

    # Model and optimiser
    model = MultivariateGaussianMixture(model_config['components'], 2)
    model.set_monitoring(os.path.abspath('runs'), 'non_monotonic_gmm')
    optimizer = torch.optim.SGD(model.parameters(), lr=model_config['learning_rate'])

    console.log(f'Model "{MODEL_NAME}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))

    # Base model from sklearn with same number of components
    base_model = GaussianMixture(n_components=model_config['components'], random_state=0).fit(features)
    base_loss = - base_model.score_samples(features).mean()
    model.set_base_loss(base_loss)

    console.log(f'Model "{MODEL_BASE_NAME}" loaded')

    status.update(status=f'Training "{MODEL_NAME}" model...')

    for it in range(model_config['iterations']):
        optimizer.zero_grad()
        loss = model(torch.from_numpy(features), it)
        loss.backward()
        optimizer.step()

    console.log(f'Model "{MODEL_NAME}" was trained successfully')
    model.clear_monitoring()

    # ===========================================================================
    #                     VISUALISE NON-MONOTONIC MODEL
    # ===========================================================================

    status.update(status=f'Visualising "{MODEL_NAME}" model...')

    grid, _, _ = model.create_grid()
    log_likelihoods = model.log_likelihoods(grid)

    model.plot_heatmap(
        model,
        features,
        os.path.abspath('out/models/nm_gmm.pdf')
    )

    model.plot_contours(
        model,
        features,
        os.path.abspath('out/models/nm_gmm_contours.pdf')
    )

console.log(f'[bold green] Experiment ran successfully!')
