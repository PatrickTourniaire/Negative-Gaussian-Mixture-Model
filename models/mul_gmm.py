import math

import torch
import torch.nn as nn
import torch.optim
import math

class GaussianMixture(nn.Module):
    def __init__(self, k_clusters, n_dim=2):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(k_clusters))
        
        self.mu = nn.Parameter(torch.empty((k_clusters, n_dim)).normal_())
        self.A = nn.Parameter(torch.empty((k_clusters, n_dim)).normal_())
        
    def forward(self, x):
        # shape x: (n_sample, 1, n_output)
        x = x.unsqueeze(1)

        latent_term = 1 / torch.sqrt(math.pow(2 * math.pi, self.n_dim) * torch.det(self.A))
        
        exp_term = torch.exp(-1/2 * torch.transpose(x - self.mu, 0, 1) * torch.inverse(self.A) * (x - self.mu))
        
        pdf = latent_term * exp_term
        
        # shape of the return tensor is (n_sample,)
        return pdf