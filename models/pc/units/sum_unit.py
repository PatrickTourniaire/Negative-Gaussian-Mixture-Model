# External imports
import torch
import numpy as np
from typing import List
from torch.nn import Module

# Local imports
from ..distributions.i_distribution import IDistribution

class SumUnit(Module):
    
    def __init__(self, inputs: List[IDistribution]):
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.inputs = inputs
        self.w = torch.ones(len(inputs), 1).type(dtype)
    
    def params(self):
        params = list(self.w)
        print(self.inputs)
        for dist in self.inputs:
            params.append(dist.A)
            params.append(dist.mu)
        
        return params

    def likelihoods(self, sample):
        return [i(sample) * w for i, w in zip(self.inputs, self.w)]

    def likelihoods_log(self, sample):
        return np.log(self.likelihoods(sample))

    def likelihoods_neglog(self, sample):
        return np.negative(self.likelihoods_log(sample))