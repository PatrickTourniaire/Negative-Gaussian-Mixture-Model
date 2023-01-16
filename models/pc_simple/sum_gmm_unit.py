from torch.nn import Module
import numpy as np
import torch

class SumGMMUnit(Module):

    def __init__(self, grid, res, sparsity=0, D=2):
        super(SumGMMUnit, self).__init__()

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
        self.w = torch.ones(D).type(dtype)
        self.mu.requires_grad, self.A.requires_grad = (
            True,
            True,
        )

    def update_covariances(self):
        """Computes the full covariance matrices from the model's parameters."""
        (M, D, _) = self.A.shape
        self.params["gamma"] = (torch.matmul(self.A, self.A.transpose(1, 2))).view(
            M, D * D
        ) / 2

    def covariances_determinants(self):
        pass

    def weights(self):
        pass

    def weights_log(self):
        pass

    def likelihoods(self, sample):
        pass

    def log_likelihoods(self, sample):
        pass

    def neglog_likelihood(self, sample):
        pass
