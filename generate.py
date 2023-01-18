# External imports
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import sys

# Local imports
from utils.pickle_handler import *

# Constants
DATA_NAME       = sys.argv[1]
PATH_DATA_PLOTS = f'out/data_plots/{DATA_NAME}.pdf'

features, clusters = make_blobs(n_samples = 100,
                  n_features = 2, 
                  centers = 1,
                  cluster_std = 0.4,
                  shuffle = True)
plt.scatter(features[:,0], features[:,1])
plt.savefig(PATH_DATA_PLOTS)

# Save pickle data
save_object(features, 'data', DATA_NAME)