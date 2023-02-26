import numpy as np
from typing import Tuple
from torch import nn

"""
NOTE:
    We might actually just use the dblquad method by scipy as it seems to work. However,
    it might be worth looking int what the difference would be for grid sampling vs doing
    the way scipy is doing.
"""

class HookPDFVerification():

    def __init__(
            self, 
            hrange: Tuple[float, float], 
            vrange: [float, float],
            hres: int,
            vres: int,
            model: nn.Module):
        # Configuration for numerical integration
        self.hrange = hrange
        self.vrange = vrange
        self.hres = hres
        self.vres = vres

        self.model = model

    
    def _create_grid(self):
        X = np.linspace(self.hrange[0], self.hrange[1], self.hres)
        Y = np.linspace(self.vrange[0], self.vrange[1], self.vres)

        return X, Y


    def _model_outputs(self, X: np.array, Y: np.array):
        pass
    
