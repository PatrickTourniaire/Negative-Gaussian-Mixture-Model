# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch import linalg

# Local imports
from .hooks import HookTensorBoard, HookVisualiseGMM

class NMMultivariateGaussianMixture(nn.Module, HookTensorBoard, HookVisualiseGMM):


    def __init__(self, n_clusters=1, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        
        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.Parameter(torch.zeros(n_clusters, n_dim, dtype=torch.float64).normal_())
        # Parameter list of random covariance matricies to be used with cholesky decomp, with shape (n_clusters, n_dim, n_dim)
        self.chol = nn.Parameter(torch.ones(n_clusters, n_dim, n_dim, dtype=torch.float64).normal_())
        # Parameter for the weights of each mixture component, which can be negative, shape (n_clusters,)
        self.weights = nn.Parameter(torch.zeros(n_clusters, dtype=torch.float64).normal_(-2,2)) 

        # Parameters which are calculated mid training, such as the cholesky decompositions
        self.params = {}


    def update_covariances(self):
        """Uses the cholesky composition to create a positive semi-definite covariance matrix
        for each cluster in range `self.n_clusters`.
        """
        self.params['sigma'] = [(self.chol[i] @ self.chol[i].T) + torch.eye(self.n_dim) for i in range(self.n_clusters)]


    def _mahalanobis(
        self, 
        X: torch.Tensor, 
        chol_sigma: torch.Tensor,
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
        return linalg.solve_triangular(chol_sigma, (X - mu).T, upper=False).T


    def _chol_covariance(self, i: int) -> torch.Tensor:
        """Retruns the covariance matrix by cholesky decomposition, expects that the
        covariances has already been updated and inserted in `self.params['sigma']`.

        Args:
            i (int): Cluster index which represents the component to compute the covar.

        Raises:
            ValueError: The covariance matrix is not positive semi-definite.

        Returns:
            torch.Tensor: Positive semi-definite covariance matrix for component `i`.
        """
        try:
            return linalg.cholesky(self.params['sigma'][i], upper=False)
        except Exception:
            try:
                return linalg.cholesky(self.params['sigma'][i] + 1e-3 * np.eye(self.n_dim), upper=False)
            except Exception:
                raise ValueError("'covars' must be symmetric, "
                                "positive-definite")


    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the log likelihood of each data point in the sample.

        Args:
            X (torch.Tensor): Sample to compute log likelihood over.

        Returns:
            torch.Tensor: Tensor with log likelihood for each data point.
        """
        self.update_covariances()

        # Cholesky decomposition for each mixture component - used as the covariance matricies
        _chol_sigma = [self._chol_covariance(i) for i in range(self.n_clusters)]

        # Log sum of determinant of each component covariance matrix
        sigma_log_det = [2 * torch.sum(torch.log(torch.diagonal(_chol_sigma[i]))) for i in range(self.n_clusters)]
        # Mahalanobis distance for each mixture component
        mahalanobis_dist = [self._mahalanobis(X, _chol_sigma[i], self.mu[i]) for i in range(self.n_clusters)]

        # Log likelihood for each mixture component
        log_likelihood = [- .5 * (torch.sum(mahalanobis_dist[i] ** 2, axis=1) + self.n_dim * np.log(2 * np.pi) + sigma_log_det[i]) for i in range(self.n_clusters)]
        # Log likelihood for each mixture component with respect to the weights
        log_likelihood = torch.logsumexp(torch.stack([log_likelihood[i] + self.weights[i] for i in range(self.n_clusters)], dim=0), dim=0)

        return log_likelihood.pow(2)
        
        
    def forward(self, X: torch.Tensor, it: int) -> torch.Tensor:
        """Computes the mean negative log likelihood to be used during training for
        gradient decent approach.

        Args:
            X (torch.Tensor): Samples to compute neg log likelihoods over.
            it (int): Iteration number, used for Tensorboard monitoring.

        Returns:
            torch.Tensor: Mean negative log likelihood over the samples.
        """
        out = self.log_likelihoods(X).mean()
        if not self.monitor: return out
        
        self.add_means(self.mu, it)
        self.add_weights(self.weights, it)
        self.add_loss(out, it)

        return out