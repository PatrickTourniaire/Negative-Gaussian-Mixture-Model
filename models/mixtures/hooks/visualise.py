import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import torch

RES = 200
VMIN = -3
VMAX = 3

class HookVisualiseGMM():

    def _create_heatmap(self, log_likelihoods: torch.Tensor):
        return (log_likelihoods.view(RES, RES).data.cpu().numpy())  

    def create_grid(self):
        ticks = np.linspace(VMIN, VMAX, RES + 1)[:-1] + 0.5 / RES
        X, Y = np.meshgrid(ticks, ticks)

        return torch.from_numpy(np.vstack((X.ravel(), Y.ravel())).T).contiguous()

    def plot (
        self,
        log_likelihoods: torch.Tensor, 
        sample: torch.Tensor, 
        save_to: str):

        heatmap = self._create_heatmap(log_likelihoods)

        scale = np.amax(heatmap[:])
        plt.imshow(
            - heatmap,
            interpolation="bilinear",
            origin="lower",
            vmin=-scale,
            vmax=scale,
            cmap=cm.RdBu,
            extent=(VMIN, VMAX, VMIN, VMAX),
        )
        plt.colorbar()

        scale = np.amax(np.abs(heatmap[:]))
        levels = np.linspace(-scale, scale, 41)

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