import math

import torch
import torch.nn as nn
import torch.optim
import math

class BivariateGaussian(nn.Module):
    # we only pass the number of clusters,
    # we have a single correlation coefficient
    # so we are restricted to 2 coordinates only
    def __init__(self, n_clusters):
        super().__init__()
        
        # Parameters of the prior distribution
        # we will use an exponential parameterization of the categorical distribution
        # so no constraint on this :)
        self.prior = nn.Parameter(torch.zeros(n_clusters))
        
        self.mean = nn.Parameter(torch.empty((1, n_clusters, 2)).normal_())
        self.log_std = nn.Parameter(torch.empty((1, n_clusters, 2)).normal_())
        
        # The correlation coefficient
        # not that we learn a reparameterization that is unconstrained,
        # and the coef is self.reparemeterized_coef.tanh(),
        # so the value is between -1 and 1 as expected
        # (one limitation is that we can't have a coef=1 or =-1, but who cares?)
        self.reparameterized_coef = nn.Parameter(torch.empty((1, n_clusters)).normal_())
        
    # return log of the pdf (i.e. marginalize over the latent variable)
    # formula is avalable here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
    # remember we are interested by the log PDF!
    def forward(self, x):
        # shape x: (n_sample, 1, n_output)
        x = x.unsqueeze(1)
        p = self.reparameterized_coef.tanh()
        # shape: (n_sample, n_latent, n_output)

        pdf = \
            - torch.log(2 * math.pi * self.log_std.exp().prod(2) * torch.sqrt(1 - p*p)) \
            - (1 / (2 * (1 - p * p))) * (
                ((x - self.mean).pow(2) / (self.log_std.exp().pow(2))).sum(2)
                - 2 * p * (x - self.mean).prod(2) / self.log_std.exp().prod(2)
            )
        
        pdf = pdf + self.prior.log_softmax(0).unsqueeze(0)
        pdf = pdf.logsumexp(dim=1)
        
        # shape of the return tensor is (n_sample,)
        return pdf