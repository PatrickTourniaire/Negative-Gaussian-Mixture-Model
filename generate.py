# External imports
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import sys

# Local imports
from utils.pickle_handler import *

# Constants
DATA_NAME       = sys.argv[1]
N_CLUSTERS      = int(sys.argv[2]) 
PATH_DATA_PLOTS = f'out/data_plots/{DATA_NAME}.pdf'

# Generate clusters
features, clusters = make_blobs(n_samples = 100,
                  n_features = 2, 
                  centers = N_CLUSTERS,
                  cluster_std = 0.4,
                  shuffle = True)

# Plot data and save to out
plt.scatter(features[:,0], features[:,1])
plt.savefig(PATH_DATA_PLOTS)

# Save pickle data
save_object(features, 'data', DATA_NAME)