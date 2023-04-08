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
from torch.utils.data import Dataset

# Local imports
from src.models.mixtures.gaussian_mixture import GaussianMixture
from src.models.mixtures.squared_gaussian_mixture import SquaredGaussianMixture
from src.models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture

from src.utils.pickle_handler import *
from src.utils.early_stopping import EarlyStopping
from src.utils.initialisation_procedures import GMMInitalisation, check_random_state

class ArtificialDataset(Dataset):
    def __init__(self, X):
        super(ArtificialDataset, self).__init__()
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

with open('sweeps/sweep_ring.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(entity="ptourniaire", project="NMMMs", config=config)

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
    'batch_size': wandb.config.batch_size,
    'sparsity': wandb.config.sparsity,
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

    tensor_train_set = torch.from_numpy(train_set).to(device) 
    tensor_val_set = torch.from_numpy(val_set).to(device) 
    tensor_test_set = torch.from_numpy(test_set).to(device)

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
    
    _weights_nmgmm = None

    if model_config['covar_shape'] == 'diag':
        _covariances_nmgmm = np.array([np.diag(np.sqrt(S)) for S in _covariances_nmgmm])
        _covariances_gmm = np.array([np.diag(S) for S in _covariances_gmm])

    
    if model_config['optimal_init'] == 'funnel' and model_config['components'] == 3:
        _means_nmgmm[0] = [3.5, 4] 
        _means_nmgmm[1] = [3.5, -4]
        _means_nmgmm[2] = [-1, 0] 

        _covariances_nmgmm[0] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[1] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[2] = [[7, 0], [0, 7]]

    
    if model_config['optimal_init'] == 'funnel' and model_config['components'] == 5:
        _means_nmgmm[0] = [3.5, 4] 
        _means_nmgmm[1] = [3.5, -4]
        _means_nmgmm[2] = [3.5, 4] 
        _means_nmgmm[3] = [3.5, -4]
        _means_nmgmm[4] = [-1, 0] 

        _covariances_nmgmm[0] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[1] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[2] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[3] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[4] = [[6, 0], [0, 6]]


    if model_config['optimal_init'] == 'mor' and model_config['components'] == 3:
        init_zip = zip(
            [torch.tensor([0, 0]), torch.tensor([1.5, 1.5]), torch.tensor([0.5, 0.5])], 
            torch.torch.from_numpy(initaliser_nmgmm.covariances_)
        )
        _covariances_nmgmm = torch.stack([torch.sqrt(torch.diag(x)) - torch.diag(i) for i, x in init_zip])
        _covariances_nmgmm = _covariances_nmgmm.cpu().numpy()

    if model_config['optimal_init'] == 'banana' and model_config['components'] == 3:
        _means_nmgmm[0] = [0, 5]
        _means_nmgmm[1] = [0, 10]
        _means_nmgmm[2] = [0, 10]

        _covariances_nmgmm[0] = [[7, 0], [0, 7]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[2] = [[2.5, 0], [0, 5]]


    if model_config['optimal_init'] == 'banana' and model_config['components'] == 4:
        _means_nmgmm[0] = [0, 5]
        _means_nmgmm[1] = [0, 10]
        _means_nmgmm[2] = [0, 10]
        _means_nmgmm[3] = [0, 5]


        _covariances_nmgmm[0] = [[7, 0], [0, 7]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[2] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[3] = [[7, 0], [0, 7]]

    
    if model_config['optimal_init'] == 'cosine' and model_config['components'] == 6:
        _means_nmgmm[0] = [0, 0.5] 
        _means_nmgmm[1] = [-1.5, 3]
        _means_nmgmm[2] = [-0.1, -3]
        _means_nmgmm[3] = [1.5, 3]
        _means_nmgmm[4] = [-3.1, -3]
        _means_nmgmm[5] = [3.1, -3] 

        _covariances_nmgmm[0] = 1.5 * np.array([[3, 0], [0, 3]])
        _covariances_nmgmm[1] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[2] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[3] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[4] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[5] = [[0.3, 0], [0, 1.5]]

    
    if model_config['optimal_init'] == 'mor' and model_config['components'] == 6:
        _means_nmgmm[0] = [0, 0] 
        _means_nmgmm[1] = [0, 0]
        _means_nmgmm[2] = [0, 0]
        _means_nmgmm[3] = [0, 0]
        _means_nmgmm[4] = [0, 0]
        _means_nmgmm[5] = [0, 0]
        _means_nmgmm[6] = [0, 0]
        _means_nmgmm[7] = [0, 0]

        _covariances_nmgmm[0] = [[3.5, 0], [0, 3.5]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 2.5]]
        _covariances_nmgmm[2] = [[1.5, 0], [0, 1.5]]
        _covariances_nmgmm[3] = [[0.5, 0], [0, 0.5]]
        _covariances_nmgmm[4] = [[2.5, 0], [0, 2.5]]
        _covariances_nmgmm[5] = [[0.5, 0], [0, 0.5]]
        _covariances_nmgmm[6] = [[0.2, 0], [0, 0.2]]
        _covariances_nmgmm[7] = [[0.1, 0], [0, 0.1]]

    
    if model_config['optimal_init'] == 'spiral' and model_config['components'] == 4:
        _means_nmgmm[0] = [0, 0] 
        _means_nmgmm[1] = [1.5, 4]
        _means_nmgmm[2] = [1.5, -2.8]
        _means_nmgmm[3] = [-2.5, 0.7]

        _covariances_nmgmm[0] = [[4, 0], [0, 4]]
        _covariances_nmgmm[1] = [[0.3, 1], [-1, 0.8]]
        _covariances_nmgmm[2] = [[1, 1], [-0.5, 0.3]]
        _covariances_nmgmm[3] = [[1, 0.5], [0.5, 0.3]]


    #=============================== NMGMM SETUP ===============================
    
    # Model and optimiser
    model = available_models[model_config['model_name']](
        device,
        n_clusters = model_config['components'], 
        n_dims = 2,
        init_means=torch.from_numpy(_means_nmgmm).to(device),
        init_sigmas=torch.from_numpy(_covariances_nmgmm).to(device),
        init_weights=_weights_nmgmm,
        sparsity=model_config['sparsity'])

    model.to(device)

    optimizer_algo = available_optimizers[model_config["optimizer"]]
    optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'])
    
    if (model_config["optimizer"] == 'sgd_mom'):
        optimizer = optimizer_algo(model.parameters(), lr=model_config['learning_rate'], momentum=wandb.config.momentum)

    console.log(f'Model "{model_config["model_name"]}" loaded with the following config:')
    console.log(json.dumps(model_config, indent=4))
    

    #============================= TRAINING NMGMM ==============================
    
    status.update(status=f'Training "{model_config["model_name"]}" model...')

    traindata = ArtificialDataset(tensor_train_set)
    valdata = ArtificialDataset(tensor_val_set)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=model_config['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(traindata, batch_size=model_config['batch_size'], shuffle=True)

    for it in range(model_config['iterations']):
        train_loss = 0
        train_total = 0

        for data in trainloader:
            optimizer.zero_grad()
            
            loss = model.neglog_likelihood(data)
            it_train_loss = model(data, it, model_config['validate_pdf'])

            train_total += 1
            train_loss += loss

            it_train_loss.backward()
            optimizer.step()
      
        val_loss = 0
        val_total = 0

        with torch.no_grad(): 
            for data in valloader:
                loss = model.neglog_likelihood(data)
                val_loss += loss
                val_total += 1
        
        model.log_means(wandb, it)
        model.log_weights(wandb, it)
        wandb.log({
            "train": {
               "loss":  train_loss / train_total,
               "it": it
            },
            "validation": {
                "loss": val_loss / val_total,
                "it": it
            }
        })


    console.log(f'Model "{model_config["model_name"]}" was trained successfully')


wandb.finish()
console.log(f'[bold green] Experiment ran successfully!')
