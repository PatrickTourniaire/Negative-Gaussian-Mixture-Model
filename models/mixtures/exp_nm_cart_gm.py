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
#  TODO: Allow for negative weights
#---------------------------------------------------------------------------

class NMGaussianMxiture(nn.Module, HookTensorBoard, HookVisualiseGMM):

    def __init__(self, n_clusters: int, n_dims: int):
        super().__init__()

        # Configurations
        self.n_dims     = n_dims
        self.n_clusters = n_clusters

        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.ParameterList(nn.Parameter(torch.rand(n_dims, dtype=torch.float64).normal_()) for _ in range(n_clusters))
        # Parameter list of random covariance matricies to be used with cholesky decomp, with shape (n_clusters, n_dim, n_dim)
        self.L = nn.ParameterList(nn.Parameter(torch.ones(n_dims, n_dims, dtype=torch.float64)) for _ in range(n_clusters))
        # Parameter for the weights of each mixture component, which can be negative, shape (n_clusters,)
        self.weights = nn.Parameter(torch.ones(n_clusters, dtype=torch.float64).normal_()) 

    def _covariance_matricies(self) -> torch.Tensor:
        self.A = [(torch.tril(self.L[i]) @ torch.tril(self.L[i]).T) + torch.eye(self.n_dims) for i in range(self.n_clusters)]
    
    def _mahalanobis_dist(
        self, 
        mu: torch.Tensor,
        Ai: torch.Tensor,
        X: torch.Tensor) -> torch.Tensor:

        return torch.Tensor([((X[i] - mu).T @ Ai @ (X[i] - mu)) for i in range(X.shape[0])])

    def _gaussian_product(self, i: int, j: int) -> torch.Tensor:
        Sigma_i, Sigma_j = torch.inverse(self.A[i]), torch.inverse(self.A[j])
        Mu_i, Mu_j       = self.mu[i], self.mu[j]

        sigma = torch.inverse(Sigma_i + Sigma_j)
        mu    = sigma @ (Sigma_i @ Mu_i + Sigma_j @ Mu_j)

        return (sigma, mu)
    
    def _product_z(self, i: int, j: int):
        Sigma_i, Sigma_j = self.A[i], self.A[j]
        Mu_i, Mu_j       = self.mu[i], self.mu[j]

        z_term = 1 / torch.sqrt(torch.det(2 * np.pi * (Sigma_i + Sigma_j)))
        mahalanobis_dist = - .5 * (Mu_i - Mu_j).T @ torch.inverse(Sigma_i + Sigma_j) @ (Mu_i - Mu_j)

        return z_term * torch.exp(mahalanobis_dist)

    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        self._covariance_matricies()

        cluster_ids   = torch.Tensor(range(self.n_clusters))
        cartesian_ids = torch.cartesian_prod(cluster_ids, cluster_ids)
        pdf = 0

        for i, j in cartesian_ids:
            i, j = int(i), int(j)
            A, mu = self._gaussian_product(i, j)

            z = 1 / torch.sqrt(np.power(2 * np.pi, self.n_dims) * torch.det(A))
            p = z * torch.exp(-.5 * self._mahalanobis_dist(mu, torch.inverse(A), X))

            #p = torch.sum(p, dim=1)

            pdf += self._product_z(i, j) * p * self.weights[i] * self.weights[j]
        
        if (pdf < 0).any(): print("INVALID PDF")
        return pdf.log()
    
    def forward(self, X: torch.Tensor, it: int):
        out = self.log_likelihoods(X).mean()
        if not self.monitor: return out
        
        self.add_means(self.mu, it)
        self.add_weights(self.weights, it)
        self.add_loss(out, it)

        return - out
