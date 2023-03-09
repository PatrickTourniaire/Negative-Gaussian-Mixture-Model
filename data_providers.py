# External imports
from matplotlib import pyplot as plt
import sys
import numpy as np

# Local imports
from src.utils.pickle_handler import *
from src.utils.datasets import *

class DataProvider():

    def __init__(self, dataset_name: str, dataset_repo: str):
        self.dataset_name = dataset_name
        self.dataset_repo = dataset_repo
    

    def create(self, dtype: str, train_size: int, val_size: int, test_size: int):
        
        seed = 12
        noise_std = 0.0
        dataset = load_data(dtype, D=2, valid_thresh=0.0, noise_std = noise_std, seed=seed, itanh=False)
        
        train_data = dataset.sample(train_size)
        val_data = dataset.sample(val_size, add_noise=True)
        test_data = dataset.sample(test_size, add_noise=True)

        # Plot data and save to out
        plt.scatter(train_data[:,0], train_data[:,1], 1, color="r", alpha=0.5)
        plt.scatter(val_data[:,0], val_data[:,1], 1, color="k", alpha=0.5)
        plt.savefig(f'out/data_plots/{self.dataset_name}.pdf')

        save_object(train_data, f'{self.dataset_repo}/train', self.dataset_name)
        save_object(val_data, f'{self.dataset_repo}/val', self.dataset_name)
        save_object(test_data, f'{self.dataset_repo}/test', self.dataset_name)


if __name__ == '__main__':
    DATA_NAME = sys.argv[1]
    DATA_TYPE = sys.argv[2]

    data_provider = DataProvider(DATA_NAME, 'data')
    data_provider.create(DATA_TYPE, 10000, 2000, 2000)