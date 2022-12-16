import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt

from models.gmm import GaussianMixture
from models.gmm_squared import GaussianMixtureSquared

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

model = GaussianMixture(3, grid, res, sparsity=3)
optimizer = torch.optim.Adam([model.A, model.w, model.mu], lr=0.1)
loss = np.zeros(501)

for it in range(501):
    optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
    cost = model.neglog_likelihood(x)  # Cost to minimize.
    cost.backward()  # Backpropagate to compute the gradient.
    optimizer.step()
    loss[it] = cost.data.cpu().numpy()

    # sphinx_gallery_thumbnail_number = 6
    if it in [0, 10, 100, 150, 250, 500]:
        plt.pause(0.01)
        plt.figure(figsize=(8, 8))
        model.plot(x)
        plt.title("Density, iteration " + str(it), fontsize=20)
        plt.axis("equal")
        plt.axis([0, 1, 0, 1])
        plt.tight_layout()
        plt.pause(0.01)
        plt.savefig(f'out/simple_gmm/simple_gmm_it{it}.pdf')