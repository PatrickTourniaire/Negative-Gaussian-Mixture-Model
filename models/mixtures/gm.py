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

    
    def __init__(self, n_clusters=1, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_clusters = n_clusters

        # Equations
        self.cholesky_comp = lambda L, D: torch.tril(L) @ torch.tril(L).T + torch.eye(D)
        self.mahalanobis   = lambda X, mu, S_i: (X - mu).T @ S_i @ (X - mu)
        
        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.ParameterList([nn.Parameter(torch.ones(n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters)])
        # Parameter list of random covariance matricies to be used with cholesky decomp, with shape (n_clusters, n_dim, n_dim)
        self.chol = nn.ParameterList([nn.Parameter(torch.ones(n_dim, n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters)])
        # Parameter for the weights of each mixture component, which can be negative, shape (n_clusters,)
        self.weights = nn.Parameter(torch.ones(n_clusters, dtype=torch.float64).normal_())

        # Parameters which are calculated mid training, such as the cholesky decompositions
        self.params = {}


    def update_covariances(self):
        """Uses the cholesky composition to create a positive semi-definite covariance matrix
        for each cluster in range `self.n_clusters`.
        """

        self.params['sigma'] = [self.cholesky_comp(self.chol[i], self.n_dim) for i in range(self.n_clusters)]


    def _mahalanobis(
        self, 
        X: torch.Tensor, 
        sigma: torch.Tensor,
        mu: torch.Tensor) -> torch.Tensor:
        """Calculates the mahalanobis distance which is a part of calculating the final pdf
        of the mixture.

        Args:
            X (torch.Tensor): Samples to compute distance over
            chol_sigma (torch.Tensor): Covraiance matrix belonging to a mixture component
            mu (torch.Tensor): Means for the corresponding mixture component

        Returns:
            torch.Tensor: The mahalanobis distance as a tensor
        """

        sqrd_mahalanobis = [self.mahalanobis(X[i], mu, torch.inverse(sigma)) for i in range(X.shape[0])]
        
        return torch.Tensor(sqrd_mahalanobis)
    
    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        self.update_covariances()
        # Softmax for creating a valid pdf 
        weights = softmax(self.weights, 0) 

        # Log sum of determinant of each component covariance matrix
        sigma_log_det = [2 * torch.log(torch.det(self.params['sigma'][i])) for i in range(self.n_clusters)]
        # Mahalanobis distance for each mixture component
        mahalanobis_dist = [self._mahalanobis(X, self.params['sigma'][i], self.mu[i]) for i in range(self.n_clusters)]

        # Log likelihood for each mixture component
        log_likelihood = [- .5 * (mahalanobis_dist[i] + self.n_dim * np.log(2 * np.pi) + sigma_log_det[i]) for i in range(self.n_clusters)]
        # Log likelihood for each mixture component with respect to the weights
        log_likelihood = torch.logsumexp(torch.stack([log_likelihood[i] + torch.log(weights[i]) for i in range(self.n_clusters)], dim=0), dim=0)

        return log_likelihood

    def forward(self, X: torch.Tensor, it: int) -> torch.Tensor:
        """Computes the mean negative log likelihood to be used during training for
        gradient decent approach.

        Args:
            X (torch.Tensor): Samples to compute neg log likelihoods over.
            it (int): Iteration number, used for Tensorboard monitoring.

        Returns:
            torch.Tensor: Mean negative log likelihood over the samples.
        """

        out = - self.log_likelihoods(X).mean()
        if not self.monitor: return out
        
        self.add_means(self.mu, it)
        self.add_weights(self.weights, it)
        self.add_loss(out, it)

        return out