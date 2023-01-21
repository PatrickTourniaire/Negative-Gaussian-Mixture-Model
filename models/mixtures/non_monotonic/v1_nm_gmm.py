import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.functional import softmax, tanh
from torch import linalg


class NMMultivariateGaussianMixture(nn.Module):
    def __init__(self, n_clusters=1, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        
        self.mu = nn.ParameterList([nn.Parameter(torch.ones(n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters)]) # list of random means
        self.chol = nn.ParameterList([nn.Parameter(torch.ones(n_dim, n_dim, dtype=torch.float64).normal_()) for _ in range(n_clusters)]) # list of random covar matrices
        self.weight = nn.Parameter(torch.zeros(n_clusters, dtype=torch.float64).random_(0,10) - 5) # random weights # random covar matrix

        # Paramters used in the PDF, calculated using the training parameters
        self.params = {}

    def update_covariances(self):
        self.params['sigma'] = [(self.chol[i] @ self.chol[i].T) + torch.eye(self.n_dim) for i in range(self.n_clusters)]

    def _mahalanobis(self, X, chol_sigma, mu):
        return linalg.solve_triangular(chol_sigma, (X - mu).T, upper=False).T

    def _chol_covariance(self, i):
        try:
            return linalg.cholesky(self.params['sigma'][i], upper=False)
        except Exception:
            try:
                return linalg.cholesky(self.params['sigma'][i] + 1e-3 * np.eye(self.n_dim), upper=False)
            except Exception:
                raise ValueError("'covars' must be symmetric, "
                                "positive-definite")

    def log_likelihoods(self, X):
        self.update_covariances()
        weights = self.weight

        _chol_sigma = [self._chol_covariance(i) for i in range(self.n_clusters)]

        sigma_log_det = [2 * torch.sum(torch.log(torch.diagonal(_chol_sigma[i]))) for i in range(self.n_clusters)]
        mahalanobis_dist = [self._mahalanobis(X, _chol_sigma[i], self.mu[i]) for i in range(self.n_clusters)]

        log_prob = [- .5 * (torch.sum(mahalanobis_dist[i] ** 2, axis=1) + self.n_dim * np.log(2 * np.pi) + sigma_log_det[i]) for i in range(self.n_clusters)]
        log_prob = torch.logsumexp(torch.stack([log_prob[i] + weights[i] for i in range(self.n_clusters)], dim=0), dim=0)

        return log_prob.pow(2)
        
    def forward(self, X):
        return self.log_likelihoods(X).mean()