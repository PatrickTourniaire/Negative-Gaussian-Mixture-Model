# External imports
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from torch.utils.tensorboard import SummaryWriter
import sys

# Local imports
from models.distributions.bivariate.v2_gmm import BivariateGaussian
from utils.pickle_handler import *

# Constants
DATA_NAME = sys.argv[1]

# Tensorboard writer
writer = SummaryWriter()

# Load data
features = load_object('data', DATA_NAME)

# Model and optimiser
model = BivariateGaussian()
optimizer = torch.optim.Adam([model.L, model.mu], lr=0.0001)

# Training iterations
for it in range(10000):
    optimizer.zero_grad()
    loss = model(torch.from_numpy(features))
    writer.add_scalar("Loss/train [v2_gmm_bivariate]", loss, it)
    loss.backward()
    optimizer.step()

writer.flush()