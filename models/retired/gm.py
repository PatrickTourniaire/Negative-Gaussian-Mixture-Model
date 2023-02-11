# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.functional import softmax
from torch import linalg

# Local imports
from .hooks import HookTensorBoard, HookVisualiseGMM

class MultivariateGaussianMixture(nn.Module, HookTensorBoard, HookVisualiseGMM):

    
    def __init__(self, n_clusters=1, n_dims=2):
        super().__init__()
        # Equations
        self.cholesky_comp = lambda L, D: torch.tril(L) @ torch.tril(L).T + torch.eye(D)
        self.mahalanobis = lambda X, mu, S_i: (X - mu).T @ S_i @ (X - mu)

        # Configurations
        self.n_dims = n_dims
        self.n_clusters = n_clusters

        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.ParameterList(
            nn.Parameter(torch.zeros(n_dims, dtype=torch.float64)) for _ in range(n_clusters))
        # Parameter list of random covariance matrices for Cholesky composition
        self.L = nn.ParameterList(
            nn.Parameter(torch.ones(n_dims, n_dims, dtype=torch.float64)) for _ in range(n_clusters))
        # Parameter for the weights of each mixture component, which can be negative, shape (n_clusters,)
        self.weights = nn.Parameter(torch.zeros(n_clusters, dtype=torch.float64))

        # Recomputed params as a result of circuit squaring
        self.params = {}


    def _cholesky_composition(self) -> torch.Tensor:
        self.sigmas = [self.cholesky_comp(self.L[i], self.n_dims) for i in range(self.n_clusters)]

    def _sqrd_mahalanobis_dist(self, mu: torch.Tensor, sigma_i: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        sqrd_mahalanobis = [self.mahalanobis(X[i], mu, sigma_i) for i in range(X.shape[0])]

        return torch.Tensor(sqrd_mahalanobis)
    
    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        self._cholesky_composition()

        self.params['weights'] = softmax(self.weights, 0) 
        pdf = torch.zeros(X.shape[0])

        for i in range(self.n_clusters):

            #z = 1 / torch.sqrt(torch.det(np.power(2 * np.pi, self.n_dims) * self.sigmas[i]))
            p = -.5 * (self.n_dims * np.log(2 * np.pi) + torch.log(torch.det(self.sigmas[i])) + self._sqrd_mahalanobis_dist(self.mu[i], torch.inverse(self.sigmas[i]), X))

            pdf += torch.exp(p + torch.log(self.params['weights'][i]))

        return pdf
    
    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(X))

    def forward(self, X: torch.Tensor, it: int) -> torch.Tensor:
        """Computes the mean negative log likelihood to be used during training for
        gradient decent approach.

        Args:
            X (torch.Tensor): Samples to compute neg log likelihoods over.
            it (int): Iteration number, used for Tensorboard monitoring.

        Returns:
            torch.Tensor: Mean negative log likelihood over the samples.
        """

        out = - self.log_likelihoods(X).logsumexp(dim=0)
        if not self.monitor: return out
        
        self.add_means(self.mu, it)
        self.add_weights(self.params['weights'], it)
        self.add_loss(out, it)

        return out