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
from torch.utils.data import Dataset

# Local imports
from src.models.mixtures.gaussian_mixture import GaussianMixture
from src.models.mixtures.squared_gaussian_mixture import SquaredGaussianMixture
from src.models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture

from src.utils.pickle_handler import *
from src.utils.early_stopping import EarlyStopping
from src.utils.nm_initialisations import create_nm_initialisation


class ArtificialDataset(Dataset):
    def __init__(self, X):
        super(ArtificialDataset, self).__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name", help="Name of experiment")
parser.add_argument("--model", help="Name of the model to experiment")
parser.add_argument("--data", help="Name of the pickle data file to train on")
parser.add_argument("--comp", help="Number of mixture components")
parser.add_argument("--it", help="Number of iterations for training")
parser.add_argument("--lr", help="Learning rate for the SGD optimiser")
parser.add_argument("--momentum", help="Momentum for SGD with momentum")
parser.add_argument("--validate_pdf", help="To use grid sampling to validate the pdf")
parser.add_argument("--optimizer", help="Which optimizer to use")
parser.add_argument("--initialisation", help="Initialisation technique to use")
parser.add_argument("--covar_shape", help="Shape of the initialised covariance matrix")
parser.add_argument("--covar_reg", help="Regularisation constant added to the digonal matrix")
parser.add_argument("--optimal_init", help="Use optimial initailisation for a particular dataset")
parser.add_argument("--batch_size", help="Specify batch size for training and validation")
parser.add_argument("--sparsity", help="Sparsity prior to regulate the weights")

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
    'covar_reg': float(args.covar_reg),
    'batch_size': int(args.batch_size),
    'optimal_init': args.optimal_init,
    'momentum': float(args.momentum),
    'sparsity': float(args.sparsity)
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

wandb.init(
    project="NMMMs",
    entity="ptourniaire",
    config={**model_config}
)

checkpoints = [i - 1 if i > 0 else i 
               for i in range(0, model_config['iterations'] + 10, 10)]

BASE_MODEL_NAME = 'sklearn_gmm'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#! Replace when running on Eddie to save artifacts on a larger disk
#OUTPUT_REPO = str(os.path.abspath('/exports/eddie/scratch/s1900878/out/'))
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

    path_sequences = f'{OUTPUT_REPO}/sequences/{args.experiment_name}'
    if not os.path.isdir(os.path.abspath(path_sequences)):
        os.makedirs(os.path.abspath(path_sequences))
    
    path_models = f'{OUTPUT_REPO}/models/{args.experiment_name}'
    if not os.path.isdir(os.path.abspath(path_models)):
        os.makedirs(os.path.abspath(path_models))
    
    path_models = f'{OUTPUT_REPO}/saved_models/'
    if not os.path.isdir(os.path.abspath(path_models)):
        os.makedirs(os.path.abspath(path_models))
    
    #=============================== INIT PARAMS ===============================

    nmgmm_params, _ = create_nm_initialisation(
        model_config['components'],
        model_config['initialisation'],
        model_config['covar_shape'],
        model_config['covar_reg'],
        train_set,
        model_config['optimal_init']
    )

    _means_nmgmm, _covariances_nmgmm, _weights_nmgmm = nmgmm_params


    #=============================== NMGMM SETUP ===============================
    
    # Model and optimiser
    model = available_models[model_config['model_name']](
        device,
        n_clusters = model_config['components'], 
        n_dims = 2,
        init_means=torch.from_numpy(_means_nmgmm),
        init_sigmas=torch.from_numpy(_covariances_nmgmm),
        init_weights=_weights_nmgmm,
        sparsity=model_config['sparsity'])

    model.to(device)
    model.set_monitoring(os.path.abspath('runs'), args.experiment_name)

    optimizer_algo = available_optimizers[model_config["optimizer"]]
    optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'])
    
    if (model_config["optimizer"] == 'sgd_mom'):
        optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'], momentum=model_config['momentum'])

    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

    console.log(f'Model "{model_config["model_name"]}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))
    

    #============================= TRAINING NMGMM ==============================
    
    status.update(status=f'Training "{model_config["model_name"]}" model...')

    train_loss_vis = []
    val_loss_vis = []

    traindata = ArtificialDataset(tensor_train_set)
    valdata = ArtificialDataset(tensor_val_set)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=model_config['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(traindata, batch_size=model_config['batch_size'], shuffle=True)

    for it in range(model_config['iterations']):        
        train_loss = 0
        train_total = 0

        train_loss_batch_vis = []

        for data in trainloader:
            optimizer.zero_grad()
            
            loss = model.neglog_likelihood(data)
            train_loss_batch_vis.append(loss)
            it_train_loss = model(data, it, model_config['validate_pdf'])
            train_total += 1
            train_loss += loss

            it_train_loss.backward()
            optimizer.step()

        val_loss = 0
        val_total = 0

        val_loss_batch_vis = []

        with torch.no_grad(): 
            
            for data in valloader:
                loss = model.neglog_likelihood(data)
                val_loss_batch_vis.append(loss)
                val_loss += loss
                val_total += 1
        
        if it in checkpoints:
            torch.save(
                model.state_dict(), 
                f'{OUTPUT_REPO}/saved_models/checkpoint{it}_{args.experiment_name}'
            )

            fig = model.plot_heatmap()
            wandb.log({f'heatmap_plot_it{it}': wandb.Image(fig)})

            fig = model.plot_contours(tensor_train_set)
            wandb.log({f'contour_plot_it{it}': wandb.Image(fig)})
            
            
        model.log_means(wandb, it)
        model.log_weights(wandb, it)
        wandb.log({
            "train": {
               "loss":  train_loss / train_total,
               "loss_batch": train_loss_batch_vis,
               "it": it
            },
            "validation": {
                "loss": val_loss / val_total,
                "loss_batch": val_loss_batch_vis,
                "it": it
            }
        })

    console.log(f'Model "{model_config["model_name"]}" was trained successfully')
    model.clear_monitoring()

    torch.save(model.state_dict(), f'{OUTPUT_REPO}/saved_models/{args.experiment_name}')


    # ===========================================================================
    #                             VISUALISE MODEL
    # ===========================================================================

    status.update(status=f'Visualising "{model_config["model_name"]}" model...')
    model_name_path = model_config["model_name"]

    

    save_object(train_loss_vis, path_models, 'train_loss_vis')
    save_object(val_loss_vis, path_models, 'val_loss_vis')

wandb.finish()
console.log(f'[bold green] Experiment ran successfully!')
