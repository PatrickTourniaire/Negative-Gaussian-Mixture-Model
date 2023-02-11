# External imports
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import sys
import numpy as np

# Local imports
from utils.pickle_handler import *

# Constants
DATA_NAME       = sys.argv[1]
N_CLUSTERS      = int(sys.argv[2])
PATH_DATA_PLOTS = f'out/data_plots/{DATA_NAME}.pdf'

def rings_sample(N, D, sigma=0.1, radia=np.array([3])):
    assert D >= 2
    
    angles = np.random.rand(N) * 2 * np.pi
    noise = np.random.randn(N) * sigma
    
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    radia_inds = np.random.choice(len(radia), N, p=weights)
    radius_samples = radia[radia_inds] + noise
    
    xs = (radius_samples) * np.sin(angles)
    ys = (radius_samples) * np.cos(angles)
    X = np.vstack((xs, ys)).T.reshape(N, 2)
    
    result = np.zeros((N, D))
    result[:, :2] = X
    if D > 2:
        result[:, 2:] = np.random.randn(N, D - 2) * sigma
    return result

# Generate clusters

"""
features, clusters = make_blobs(n_samples = 100,
                  n_features = 2, 
                  centers = N_CLUSTERS,
                  cluster_std = 0.1,
                  shuffle = True)
"""

features = rings_sample(100, 2)

# Plot data and save to out
plt.scatter(features[:,0], features[:,1])
plt.savefig(PATH_DATA_PLOTS)

# Save pickle data
save_object(features, 'data', DATA_NAME)