import math
from torch import linalg
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import math

class MultivariateGaussian(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        
        self.mu = nn.Parameter(torch.ones(n_dim, dtype=torch.float64).normal_()) # random mean
        self.chol = nn.Parameter(torch.ones(n_dim, n_dim, dtype=torch.float64).normal_()) # random covar matrix

        self.params = {}

    def update_covariances(self):
        self.params['sigma'] = (self.chol @ self.chol.T) + torch.eye(self.n_dim)

    def _mahalanobis(self, X, chol_sigma):
        return linalg.solve_triangular(chol_sigma, (X - self.mu).T, upper=False).T

    def _chol_covariance(self):
        try:
            return linalg.cholesky(self.params['sigma'], upper=False)
        except Exception:
            try:
                return linalg.cholesky(self.params['sigma'] + 1e-3 * np.eye(self.n_dim), upper=False)
            except Exception:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
        
    def forward(self, X):
        self.update_covariances()

        _chol_sigma = self._chol_covariance()

        sigma_log_det = 2 * torch.sum(torch.log(torch.diagonal(_chol_sigma)))
        mahalanobis_dist = self._mahalanobis(X, _chol_sigma)

        log_prob = - .5 * (torch.sum(mahalanobis_dist ** 2, axis=1) + self.n_dim * np.log(2 * np.pi) + sigma_log_det)

        return - torch.mean(log_prob)