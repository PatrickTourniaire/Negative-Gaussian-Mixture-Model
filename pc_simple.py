import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt

from models.pc_simple import SumGMMUnit

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
torch.manual_seed(0)
N = 10000  # Number of samples
t = torch.linspace(0, 2 * np.pi, N + 1)[:-1]
x = torch.stack((0.5 + 0.4 * (t / 7) * t.cos(), 0.5 + 0.3 * t.sin()), 1)
x = x + 0.02 * torch.randn(x.shape)
x = x.type(dtype)
x.requires_grad = True

# Create a uniform grid on the unit square:
res = 200
ticks = np.linspace(0, 1, res + 1)[:-1] + 0.5 / res
X, Y = np.meshgrid(ticks, ticks)

grid = torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous().type(dtype)

model = SumGMMUnit(grid, res)