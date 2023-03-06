# External imports
import numpy as np
import torch
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


class BaseHookVisualise():


    def _confidence_ellipse(self, ax, sigma, mu, n_std=2.3, facecolor='none', **kwargs):
        pearson = sigma[0][1]/np.sqrt(sigma[0][0] * sigma[1][1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(sigma[0][0]) * n_std
        mean_x = mu[0]

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(sigma[1][1]) * n_std
        mean_y = mu[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)


    def get_grid(self, r, i, j, cond):
        grid = np.meshgrid(r,r)

        grid = np.stack(grid,2)
        grid = grid.reshape(-1,2)
        
        num_point = len(grid)
        grid_cond = np.tile(cond[None,:], [num_point, 1])
        
        grid_cond[:,i] = grid[:,0]
        grid_cond[:,j] = grid[:,1]

        return grid_cond