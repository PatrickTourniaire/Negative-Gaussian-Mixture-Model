import torch
import torch.nn as nn
import numpy as np

class BivariateGaussian(nn.Module):
    def __init__(self):
        super(BivariateGaussian, self).__init__()
        self.mu = nn.Parameter(torch.ones(2, dtype=torch.float64).normal_()) # random mean
        self.L = nn.Parameter(torch.ones(2, 2, dtype=torch.float64).normal_()) # random covar matrix
    
    def forward(self, x):
        N, D = x.shape
        x_mu = x - self.mu
        sigma = self.L @ self.L.T + torch.eye(2)
        
        log_likelihood = 0.5 * (N * D * np.log(2 * np.pi) + N * 2. * torch.sum(torch.log(torch.diag(sigma))) + torch.sum((x_mu @ torch.inverse(sigma))**2, dim=-1))
        loss = -torch.mean(log_likelihood)
        return loss