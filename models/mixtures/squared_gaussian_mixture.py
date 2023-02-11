# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
import matplotlib.cm as cm

# Local imports
from .hooks import HookTensorBoard, BaseHookVisualise


# Equations
cholesky_comp = lambda L, D: torch.tril(L) @ torch.tril(L).t() + torch.eye(D)
mahalanobis   = lambda x, mu, S_inv: (x - mu).t() @ S_inv @ (x - mu)


class SquaredGaussianMixture(nn.Module, HookTensorBoard, BaseHookVisualise):

    def __init__(self, n_clusters: int, n_dims: int):
        super(SquaredGaussianMixture, self).__init__()

        # Configurations
        self.n_dims = n_dims
        self.n_clusters = n_clusters

        # Parameters of component means (n_clusters, n_dim)
        self.means = nn.Parameter(torch.rand(n_clusters, n_dims, dtype=torch.float64).normal_())
        # Parameters of matrices used for Cholesky composition (n_clusters, n_dim, n_dim)
        self.chols = nn.Parameter(torch.rand(n_clusters, n_dims, n_dims, dtype=torch.float64).normal_())
        # Parameter for the weights of each mixture component (n_clusters,)
        self.weights = nn.Parameter(torch.rand(n_clusters, dtype=torch.float64).normal_())

        # Parameters used for tensorboard
        self.tb_params = {}
    

    #===================================================================================================
    #                                       INTERNAL COMPUTATIONS
    #===================================================================================================

    def _chol_composition(self) -> torch.Tensor:
        self.sigmas = [cholesky_comp(self.chols[i], self.n_dims) for i in range(self.n_clusters)]
    

    def _sqrd_mahalanobis(self, X: torch.Tensor, S_inv: torch.tensor, mu: torch.Tensor):
        sqrd_mahalanobis = [mahalanobis(X[i], mu, S_inv) for i in range(X.shape[0])]
        return torch.stack(sqrd_mahalanobis, dim=0)
    

    def _sqrd_params(self, i: int, j: int) -> torch.Tensor:
        sigma_i, sigma_j = torch.inverse(self.sigmas[i]), torch.inverse(self.sigmas[j])
        mu_i, mu_j = self.means[i], self.means[j]

        # Compute the squared Gaussian params
        sigma = torch.inverse(sigma_i + sigma_j)
        mu = sigma @ (sigma_i @ mu_i + sigma_j @ mu_j)

        return (sigma, mu)

    def _squared_norm_term(self, i: int, j: int):
        sigma_i, sigma_j = self.sigmas[i], self.sigmas[j]
        mu_i, mu_j = self.means[i], self.means[j]

        z_term = 1 / torch.sqrt(torch.det(2 * np.pi * (sigma_i + sigma_j)))
        mahalanobis_dist = - .5 * (mu_i - mu_j).T @ torch.inverse(sigma_i + sigma_j) @ (mu_i - mu_j)

        return z_term * torch.exp(mahalanobis_dist)
    

    def _grid_validation(self):
        grid, _, _ = self.create_grid()
        pdf_out = self.pdf(grid)

        if ((pdf_out < 0).any()):
            raise ValueError("The model is not a valid PDF!")


    #===================================================================================================
    #                                        LIKELIHOOD METHODS
    #===================================================================================================

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        self._chol_composition()

        # Lambda expressions for calculating the PDF
        norm_constant    = lambda S: torch.sqrt(np.power(2 * np.pi, self.n_dims) * torch.det(S))
        exponential      = lambda S, mu: - .5 * self._sqrd_mahalanobis(X, torch.inverse(S), mu)
        density_function = lambda S, mu: (1 / norm_constant(S)) * torch.exp(exponential(S, mu))

        cluster_ids = torch.Tensor(range(self.n_clusters))
        cartesian_ids = torch.cartesian_prod(cluster_ids, cluster_ids)
        cartesian_ids = cartesian_ids.data.cpu().numpy().astype(int)
        
        weights = [self.weights[i] * self.weights[j] for (i, j) in cartesian_ids]
        weights = softmax(torch.stack(weights, dim=0), dim=0)

        component_likelihoods = []
        normalisers = []
        tb_means = []
        tb_sigmas = []

        for (k, (i, j)) in enumerate(cartesian_ids):
            sigma, means = self._sqrd_params(i, j)
            tb_means.append(means)
            tb_sigmas.append(sigma)

            likelihood = density_function(sigma, means)
            normalisers.append(self._squared_norm_term(i, j) * weights[k])
            
            component_likelihoods.append(weights[k] * likelihood)
        
        self.tb_params['means'] = torch.stack(tb_means, dim=0)
        self.tb_params['sigmas'] = torch.stack(tb_sigmas, dim=0)
        self.tb_params['weights'] = weights

        return (torch.stack(normalisers, dim=0).sum(dim=0)) * torch.stack(component_likelihoods, dim=0).sum(dim=0)
    

    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(X))
    

    def forward(self, X: torch.Tensor, it: int, validate: bool = False) -> torch.Tensor:
        if validate and (it % 100 == 0): self._grid_validation()

        out = - (self.log_likelihoods(X).logsumexp(dim=0) / X.shape[0])
        if not self.monitor: return out
        
        self.add_means(self.tb_params['means'] , it)
        self.add_weights(self.tb_params['weights'], it)
        self.add_loss(out, it)

        return out
    

    #===================================================================================================
    #                                       VISUALISATION METHODS
    #===================================================================================================

    def plot_contours(self, samples: torch.Tensor, save_to: str):
        _, ax = plt.subplots()
        x, y = samples[:,0], samples[:,1]
        ax.scatter(x, y, s=0.5)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        for i in range(self.n_clusters):
            sigma = self.tb_params['sigmas'][i].data.cpu().numpy()
            mu = self.tb_params['means'][i].data.cpu().numpy()
            
            colour = 'blue' if self.tb_params['weights'][i] > 0 else 'red'
            config = {
                'edgecolor': colour,
                'facecolor': colour,
                'alpha': .2
            }

            self._confidence_ellipse(ax, sigma, mu, **config)
            ax.scatter(mu[0], mu[1], c='red', s=3)
        
        ax.set_title('Monotonic Gaussian Mixture')
        plt.savefig(save_to)


    def plot_heatmap(self, samples: torch.Tensor, save_to: str):
        grid, _, _ = self.create_grid()
        log_likelihoods = self.pdf(grid)

        heatmap = self._create_heatmap(log_likelihoods)

        plt.imshow(
            - heatmap,
            interpolation="bilinear",
            origin="lower",
            vmin=0,
            vmax=log_likelihoods.max(),
            cmap=cm.RdBu,
            extent=(5, 12, -10, 0),
        )
        plt.colorbar()

        levels = np.linspace(-10, 0, 41)
        plt.contour(
            heatmap,
            origin="lower",
            linewidths=1.0,
            colors="#C8A1A1",
            levels=levels,
            extent=(self.vmin, self.vmax, self.vmin, self.vmin),
        )

        plt.scatter(samples[:, 0], samples[:, 1], color="k")
        plt.savefig(save_to)