import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

RES = 200
VMIN = -4
VMAX = 4

class HookVisualiseGMM():

    def _create_heatmap(self, log_likelihoods: torch.Tensor):
        return (log_likelihoods.view(RES, RES).data.cpu().numpy()) 
    
    def _confidence_ellipse(self, ax, sigma, mu, n_std=2.3, facecolor='none', **kwargs):
        U, s, Vt = np.linalg.svd(sigma)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)

        for nsig in range(1, 4):
            ax.add_patch(Ellipse((mu[0], mu[1]), nsig * width, nsig * height,
                                angle, **kwargs))

    def create_grid(self):
        ticks = np.linspace(VMIN, VMAX, RES + 1)[:-1] + 0.5 / RES
        X, Y = np.meshgrid(ticks, ticks)

        return torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous(), X, Y

    def plot_heatmap(
        self,
        model: nn.Module,
        sample: torch.Tensor, 
        save_to: str):

        grid, X, Y = model.create_grid()
        log_likelihoods = model.log_likelihoods(grid)

        heatmap = self._create_heatmap(log_likelihoods)

        scale = np.amax(heatmap[:])
        plt.imshow(
            - heatmap,
            interpolation="bilinear",
            origin="lower",
            vmin=-10,
            vmax=0,
            cmap=cm.RdBu,
            extent=(VMIN, VMAX, VMIN, VMAX),
        )
        plt.colorbar()

        scale = np.amax(np.abs(heatmap[:]))
        levels = np.linspace(-10, 0, 41)

        plt.contour(
            heatmap,
            origin="lower",
            linewidths=1.0,
            colors="#C8A1A1",
            levels=levels,
            extent=(VMIN, VMAX, VMIN, VMAX),
        )

        plt.scatter(sample[:, 0], sample[:, 1], color="k")
        plt.savefig(save_to)
    
    def plot_contours(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        save_to: str):

        _, ax = plt.subplots()
        x, y = sample[:,0], sample[:,1]
        ax.scatter(x, y, s=0.5)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        for i in range(model.n_clusters):
            sigma = model.A[i].data.cpu().numpy()
            mu = model.mu[i].data.cpu().numpy()
            colour = 'blue' if model.weights[i] > 0 else 'red'

            self._confidence_ellipse(ax, sigma, mu, edgecolor = colour, facecolor = colour, alpha = .1)

            ax.scatter(mu[0], mu[1], c='red', s=3)
        
        ax.set_title('Non-Monotonic Gaussian Mixture')
        plt.savefig(save_to)