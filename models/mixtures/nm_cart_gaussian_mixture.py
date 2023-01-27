# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch import linalg
from torch.nn.functional import softmax
from rich import print
import math

# Local imports
from .hooks import HookTensorBoard, HookVisualiseGMM

class NMMultivariateGaussianMixture(nn.Module, HookTensorBoard, HookVisualiseGMM):


    def __init__(self, n_clusters=1, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        
        # Parameter list of random means with shape (n_clusters, n_dim)
        self.mu = nn.ParameterList(nn.Parameter(torch.zeros(n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters))
        # Parameter list of random covariance matricies to be used with cholesky decomp, with shape (n_clusters, n_dim, n_dim)
        self.chol = nn.ParameterList(nn.Parameter(torch.ones(n_dim, n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters))
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
    
    def _gaussian_product(self, i: int, j: int) -> torch.Tensor:
        Sigma_i, Sigma_j = self._chol_covariance(i), self._chol_covariance(j)
        Mu_i, Mu_j       = self.mu[i], self.mu[j]

        # Gaussian product formulation
        sigma = torch.inverse(torch.inverse(Sigma_i) + torch.inverse(Sigma_j))
        mu    = sigma @ (torch.inverse(Sigma_i) @ Mu_i + torch.inverse(Sigma_j) @ Mu_j)

        return (sigma, mu)
    
    def _product_z(self, i: int, j: int):
        Sigma_i, Sigma_j = self._chol_covariance(i), self._chol_covariance(j)
        Mu_i, Mu_j       = self.mu[i], self.mu[j]

        z_term = 1 / torch.sqrt(torch.det(2 * np.pi * (Sigma_i + Sigma_j)))
        mahalanobis_dist = - .5 * (Mu_i - Mu_j).T @ torch.inverse(Sigma_i + Sigma_j) @ (Mu_i - Mu_j)

        return z_term * torch.exp(mahalanobis_dist)


    def _unormalised_pdf(self, simga: torch.Tensor, mu: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Log sum of determinant of each component covariance matrix
        z_term = 1 / torch.sqrt(np.power(2 * np.pi, self.n_dim) * torch.det(simga)) # ! Might not be right to use diagonal sum
        # Mahalanobis distance for each mixture component
        mahalanobis_dist = self._mahalanobis(X, simga, mu)

        # Log likelihood for each mixture component
        pdf =  z_term * torch.exp(mahalanobis_dist)
        
        return pdf

    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        self.update_covariances()
        weights = softmax(self.weights, dim=0)

        cluster_ids   = torch.Tensor(range(self.n_clusters))
        cartesian_ids = torch.cartesian_prod(cluster_ids, cluster_ids)
        
        log_likelihood = 0

        for i, j in cartesian_ids:
            i, j = int(i), int(j)
            sigma, mu = self._gaussian_product(i, j)
            pdf = self._product_z(i, j) * self._unormalised_pdf(sigma, mu, X)
            if (pdf > 1).any() or (pdf < 0).any(): print(pdf)
            log_likelihood += pdf * weights[i] * weights[j]

        return torch.log(log_likelihood).mean()
        
        
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

        return - out