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
from models.distributions.multivariate.v3_gmm import MultivariateGaussian
from utils.pickle_handler import *

def plot_gaussian_contours(mean, cov, xy_points):
    x, y = np.linspace(-10, -6, 100), np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    

    sigma_tmp = (cov @ cov.T + torch.eye(2))
    sigma_tmp = sigma_tmp.data.cpu().numpy()
    
    mean_tmp = mean.data.cpu().numpy()
    
    rv = multivariate_normal(mean_tmp, sigma_tmp)
    plt.contour(X, Y, rv.pdf(pos))
    plt.scatter(xy_points[:, 0], xy_points[:, 1])
    
    return plt


# Constants
DATA_NAME = sys.argv[1]

# Tensorboard writer
writer = SummaryWriter()

# Load data
features = load_object('data', DATA_NAME)

# Model and optimiser
model = MultivariateGaussian()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

gm = GaussianMixture(n_components=1, random_state=0).fit(features)
gm_score = -gm.score_samples(features).mean()

# Training iterations
for it in range(5000):
    optimizer.zero_grad()
    loss = model(torch.from_numpy(features))
    writer.add_scalars(f'Loss/train', {
        'v3_GMM_multivariate': loss,
        'GMM_MLE': gm_score,
    }, it)
    writer.add_scalars(f'Means/train', {
        'x': model.mu[0],
        'y': model.mu[1]
    }, it)
    loss.backward()
    optimizer.step()

plt = plot_gaussian_contours(model.mu, model.params['sigma'], features)
plt.savefig(f'out/models/{DATA_NAME}.pdf')

writer.flush()