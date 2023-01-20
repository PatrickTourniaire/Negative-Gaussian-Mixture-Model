# External imports
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# Local imports
from models.mixtures.v1_gmm_mixture import MultivariateGaussianMixture
from utils.pickle_handler import *

# Constants
DATA_NAME = sys.argv[1]
N_CLUSTERS = int(sys.argv[2])

def plot_gaussian_contours(plt, mean, cov, xy_points):
    x, y = np.linspace(1, 8, 100), np.linspace(-5, -11, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    sigma_tmp = cov.data.cpu().numpy()
    mean_tmp = mean.data.cpu().numpy()
    
    rv = multivariate_normal(mean_tmp, sigma_tmp)
    plt.contour(X, Y, rv.pdf(pos))
    plt.scatter(xy_points[:, 0], xy_points[:, 1])
    
    return plt

# Tensorboard writer
writer = SummaryWriter()

# Load data
features = load_object('data', DATA_NAME)

# Model and optimiser
model = MultivariateGaussianMixture(N_CLUSTERS, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

gm = GaussianMixture(n_components=N_CLUSTERS, random_state=0).fit(features)
gm_score = -gm.score_samples(features).mean()

# Training iterations
for it in range(5000):
    optimizer.zero_grad()
    loss = model(torch.from_numpy(features))

    # Plot loss graph with the loss with MLE for comparison
    writer.add_scalars(f'Loss/train', {
        'GMM_SGD': loss,
        'GMM_MLE': gm_score,
    }, it)

    # Plot the means with respect to the clusters
    for i_mu in range(N_CLUSTERS):
        writer.add_scalars(f'Means/train', {
            f'X_SGD_{i_mu}': model.mu[i_mu][0],
            f'Y_SGD_{i_mu}': model.mu[i_mu][1],
            f'X_MLE_{i_mu}': gm.means_[i_mu][0],
            f'Y_MLE_{i_mu}': gm.means_[i_mu][1]
        }, it)

    loss.backward()
    optimizer.step()

writer.flush()
