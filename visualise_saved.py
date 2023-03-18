import glob
import torch
import os

from src.utils.pickle_handler import *
from src.models.mixtures.squared_nm_gaussian_mixture import NMSquaredGaussianMixture

paths = glob.glob(f'out/saved_models/*', recursive=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for path in paths:
    dataset = path.split('/')[-1].split('_')[2]

    train_set = load_object('data/train', dataset)
    val_set = load_object('data/val', dataset)
    test_set = load_object('data/test', dataset)

    tensor_train_set = torch.from_numpy(train_set) if str(device) == 'cpu' else torch.from_numpy(train_set).cuda()
    tensor_val_set = torch.from_numpy(val_set) if str(device) == 'cpu' else torch.from_numpy(val_set).cuda()
    tensor_test_set = torch.from_numpy(test_set) if str(device) == 'cpu' else torch.from_numpy(test_set).cuda()
    
    path_sequences = f'out/models/{path.split("/")[-1]}'
    
    comps = int(path.split('/')[-1].split('_')[3].split('comp')[-1])
    #lr = float(path.split('/')[-1].split('_')[4].split('lr')[-1])

    try:
        model = NMSquaredGaussianMixture(comps, 2)
        model.load_state_dict(state_dict=torch.load(path, map_location=torch.device('cpu')))
        model.neglog_likelihood(tensor_train_set)
        model.sequence_visualisation(
            tensor_train_set,
            tensor_val_set,
            os.path.abspath(f"{path_sequences}.png")
        )
    except Exception as e:
        print(e)