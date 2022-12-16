# External imports
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Module
from torch.nn.functional import softmax, log_softmax
from pykeops.torch import Vi, Vj

# Local imports
from i_distributions import IDistribution

class MultivariateGaussian(IDistribution, Module):

    def __init__(self, grid, res, sparsity=0, D=2):
        super(MultivariateGaussian, self).__init__()

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.params = {}
        # We initialize our model with random blobs scattered across
        # the unit square, with a small-ish radius:
        self.mu = torch.rand(D).type(dtype)
        self.A = 15 * torch.ones(D, D) * torch.eye(D, D)
        self.A = (self.A).type(dtype).contiguous()
        self.sparsity = sparsity
        self.grid = grid
        self.res = res
        self.mu.requires_grad, self.A.requires_grad = (
            True,
            True,
        )
    
    #===========================================================================
    #                            Local Methods
    #===========================================================================
    def _compute_precision(self):
        return self.A * self.A.T

    #===========================================================================
    #                            Public Methods
    #===========================================================================
    def likelihood(self, sample):
        # Computes the precision matrix - inverse of covar
        pres = self.compute_precision()

        z   = 1 / torch.sqrt((2 * torch.pi)**self.D * torch.det(torch.inverse(pres)))
        exp = torch.exp(-0.5 * torch.sum((sample - self.mu) * torch.mv(pres, (sample - self.mu))))
        
        return z * exp 

    def likelihood_log(self, sample):
        return torch.log(self.likelihood(sample))
    
    def likelihood_neglog(self, sample):
        return (-1) * self.likelihood_log(sample) 