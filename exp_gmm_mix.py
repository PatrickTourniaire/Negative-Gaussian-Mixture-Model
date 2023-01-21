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
from models.mixtures.non_monotonic.v1_nm_gmm import NMMultivariateGaussianMixture
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
model = NMMultivariateGaussianMixture(N_CLUSTERS, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

gm = GaussianMixture(n_components=N_CLUSTERS, random_state=0).fit(features)
gm_score = gm.score_samples(features).mean()

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
    for i in range(N_CLUSTERS):
        writer.add_scalars(f'Means/train', {
            f'X_SGD_{i}': model.mu[i][0],
            f'Y_SGD_{i}': model.mu[i][1],
        }, it)

        writer.add_scalars(f'Weights/train', {
            f'w_{i}': model.weight[i]
        })


    loss.backward()
    optimizer.step()

res = 200
ticks = np.linspace(-3, 3, res + 1)[:-1] + 0.5 / res
X, Y = np.meshgrid(ticks, ticks)

grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous()

# Heatmap:
heatmap = model.log_likelihoods(grid)
heatmap = (
    heatmap.view(res, res).data.cpu().numpy()
)  # reshape as a "background" image

scale = np.amax(heatmap[:])
plt.imshow(
    -heatmap,
    interpolation="bilinear",
    origin="lower",
    vmin=-scale,
    vmax=scale,
    cmap=cm.RdBu,
    extent=(-3, 3, -3, 3),
)
plt.colorbar()

log_heatmap = - model.log_likelihoods(grid)
log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()

scale = np.amax(np.abs(log_heatmap[:]))
levels = np.linspace(-scale, scale, 41)

plt.contour(
    log_heatmap,
    origin="lower",
    linewidths=1.0,
    colors="#C8A1A1",
    levels=levels,
    extent=(-3, 3, -3, 3),
)
plt.scatter(features[:, 0], features[:, 1], color="k")

plt.savefig('out/models/nm_gmm.pdf')

writer.flush()
