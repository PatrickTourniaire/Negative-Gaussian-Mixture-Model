# External imports
import torch
import sys
import os
from sklearn.mixture import GaussianMixture

# Local imports
from models.mixtures.nm_gaussian_mixture import NMMultivariateGaussianMixture
from utils.pickle_handler import *

# Constants
DATA_NAME = sys.argv[1]
N_CLUSTERS = int(sys.argv[2])

# Load data - which is generated with `generate.py`
features = load_object('data', DATA_NAME)

#===========================================================================
#                   TRAIN AND MONITOR WITH TENSORBOARD
#===========================================================================

# Model and optimiser
model = NMMultivariateGaussianMixture(N_CLUSTERS, 2)
model.set_monitoring(os.path.abspath('runs'), 'non_monotonic_gmm')
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Base model from sklearn with same number of components
base_model = GaussianMixture(n_components=N_CLUSTERS, random_state=0).fit(features)
base_loss = base_model.score_samples(features).mean()
model.set_base_loss(base_loss)

# Training iterations
for it in range(5000):
    optimizer.zero_grad()
    loss = model(torch.from_numpy(features), it)
    loss.backward()
    optimizer.step()

model.clear_monitoring()

#===========================================================================
#                     VISUALISE NON-MONOTONIC MODEL
#===========================================================================

grid, _, _ = model.create_grid()
log_likelihoods = model.log_likelihoods(grid)

model.plot_heatmap(
    model,
    features,
    os.path.abspath('out/models/nm_gmm.pdf')
)

model.plot_contours(
    model,
    features,
    os.path.abspath('out/models/nm_gmm_contours.pdf')
)