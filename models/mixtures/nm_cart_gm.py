# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch import linalg
from torch.nn.functional import softmax
from rich import print
import math

from pykeops.torch import Vi, Vj, LazyTensor

# Local imports
from .hooks import HookTensorBoard, HookVisualiseGMM

#---------------------------------------------------------------------------
#  todo                             TODO
#  ///TODO: Re-implement the Gaussian mixture
#  ///TODO: Extend for cartesian product of the PC
#  ///TODO: Allow for negative weights
#  TODO: Perform more extensive test - so far looking good
#  TODO: Modify to propegate the norm const to sum unit
#  ///TODO: Modify to log new sigmas and mus (squaring)
#  TODO: Modify to visualise new components after squaring
#---------------------------------------------------------------------------

class NMGaussianMxiture(nn.Module, HookTensorBoard, HookVisualiseGMM):

    def __init__(self, n_clusters: int, n_dims: int):
        super().__init__()

        # Equations
        self.cholesky_comp = lambda L, D: torch.tril(L) @ torch.tril(L).T + torch.eye(D)
        self.mahalanobis   = lambda X, mu, S_i: (X - mu).T @ S_i @ (X - mu)

        # Configurations
        self.n_dims     = n_dims
        self.n_clusters = n_clusters

        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.ParameterList(nn.Parameter(torch.zeros(n_dims, dtype=torch.float64)) for _ in range(n_clusters))
        # Parameter list of random covariance matricies to be used with cholesky decomp, with shape (n_clusters, n_dim, n_dim)
        self.L = nn.ParameterList(nn.Parameter(torch.ones(n_dims, n_dims, dtype=torch.float64)) for _ in range(n_clusters))
        # Parameter for the weights of each mixture component, which can be negative, shape (n_clusters,)
        self.weights = nn.Parameter(torch.ones(n_clusters, dtype=torch.float64).normal_())

        # Recomputed params as a result of circuit squaring
        self.params = {}

    def _cholesky_composition(self) -> torch.Tensor:
        self.sigmas = [self.cholesky_comp(self.L[i], self.n_dims) for i in range(self.n_clusters)]
    

    def _sqrd_mahalanobis_dist(self, mu: torch.Tensor, sigma_i: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        sqrd_mahalanobis = [self.mahalanobis(X[i], mu, sigma_i) for i in range(X.shape[0])]
        
        return torch.Tensor(sqrd_mahalanobis)


    def _squared_params(self, i: int, j: int) -> torch.Tensor:
        sigma_i, sigma_j = torch.inverse(self.sigmas[i]), torch.inverse(self.sigmas[j])
        mu_i, mu_j       = self.mu[i], self.mu[j]

        # Compute the squared gaussian params
        sigma = torch.inverse(sigma_i + sigma_j)
        mu    = sigma @ (sigma_i @ mu_i + sigma_j @ mu_j)

        return (sigma, mu)
    

    def _squared_norm_term(self, i: int, j: int):
        sigma_i, sigma_j = self.sigmas[i], self.sigmas[j]
        mu_i, mu_j       = self.mu[i], self.mu[j]

        z_term = 1 / torch.sqrt(torch.det(2 * np.pi * (sigma_i + sigma_j)))
        mahalanobis_dist = - .5 * (mu_i - mu_j).T @ torch.inverse(sigma_i + sigma_j) @ (mu_i - mu_j)

        return z_term * torch.exp(mahalanobis_dist)
    

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        self._cholesky_composition()

        cluster_ids   = torch.Tensor(range(self.n_clusters))
        cartesian_ids = torch.cartesian_prod(cluster_ids, cluster_ids)
        
        pdf = 0
        norm = 0

        for i, j in cartesian_ids:
            i, j = int(i), int(j)
            sigma, mu = self._squared_params(i, j)

            self.params[(i,j)] = (sigma, mu, self.weights[i] * self.weights[j])

            z = 1 / torch.sqrt(torch.det(np.power(2 * np.pi, self.n_dims) * sigma))
            p = z * torch.exp(-.5 * self._sqrd_mahalanobis_dist(mu, torch.inverse(sigma), X))

            pdf +=  p * self.weights[i] * self.weights[j]
            norm += self._squared_norm_term(i, j) * self.weights[i] * self.weights[j]
        
        return z * pdf


    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        return self.pdf(X).log()
    

    def forward(self, X: torch.Tensor, it: int):
        out = self.log_likelihoods(X).mean()
        if not self.monitor: return out

        mu =  [mu for (_, mu, _) in list(self.params.values())]
        weights =  [w for (_, _, w) in list(self.params.values())]
        
        self.add_means(mu, it)
        self.add_weights(weights, it)
        self.add_loss(out, it)

        return - out
