# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.functional import softmax
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import integrate

# Local imports
from .hooks import HookTensorBoard, BaseHookVisualise


# Equations
cholesky_comp = lambda L, D: L @ L.t() + torch.eye(D)
mahalanobis   = lambda x, mu, S_inv: (x - mu).t() @ S_inv @ (x - mu)

def _batch_mahalanobis(bL, bx):
    """
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


class GaussianMixture(nn.Module, HookTensorBoard, BaseHookVisualise):

    def __init__(self, n_clusters: int, n_dims: int, device: str):
        super(GaussianMixture, self).__init__()
        self.device = device

        # Configurations
        self.n_dims = n_dims
        self.n_clusters = n_clusters

        # Parameters of component means (n_clusters, n_dim)
        self.means = nn.Parameter(torch.rand(n_clusters, n_dims, dtype=torch.float64).normal_())
        # Parameters of matrices used for Cholesky composition (n_clusters, n_dim, n_dim)
        self.chols = nn.Parameter(torch.rand(n_clusters, n_dims, n_dims, dtype=torch.float64).normal_())
        # Parameter for the weights of each mixture component (n_clusters,)
        self.weights = nn.Parameter(torch.rand(n_clusters, dtype=torch.float64).normal_())

        # Parameters used for tensorboard
        self.tb_params = {}
    

    #===================================================================================================
    #                                       INTERNAL COMPUTATIONS
    #===================================================================================================

    def _chol_composition(self) -> torch.Tensor:
        self.sigmas = [cholesky_comp(self.chols[i], self.n_dims) for i in range(self.n_clusters)]
    

    def _sqrd_mahalanobis(self, X: torch.Tensor, S_inv: torch.tensor, mu: torch.Tensor):
        sqrd_mahalanobis = [mahalanobis(X[i], mu, S_inv) for i in range(X.shape[0])]
        return torch.stack(sqrd_mahalanobis, dim=0)
    

    def _samplerange(self, samples: torch.Tensor):
        x, y = samples[:,0], samples[:,1]

        return (x.min(), x.max(), y.min(), y.max())


    def _validation(self, samples: torch.Tensor, it: int):
        x_pad, y_pad = 10, 10
        x_min, x_max, y_min, y_max = self._samplerange(samples)

        f = lambda x, y: self.pdf(torch.Tensor([[x, y]])).data.cpu().numpy().astype(float)[0]
        integral, _ = integrate.dblquad(f, x_min-x_pad, x_max+x_pad, y_min-y_pad, y_max+y_pad)

        self.add_integral(torch.Tensor([integral]), it)


    #===================================================================================================
    #                                        LIKELIHOOD METHODS
    #===================================================================================================

    def pdf(self, X: torch.Tensor) -> torch.Tensor:
        self._chol_composition()

        weights = softmax(self.weights, dim=0)
        # Add normalised weights to print params
        self.tb_params['weights'] = weights

        # Compute PDF for each cluster by weighted sum
        norm_constant    = lambda S: torch.sqrt(np.power(2 * np.pi, self.n_dims) * torch.det(S))
        exponential      = lambda L, mu: - .5 * _batch_mahalanobis(L, (X - mu))
        density_function = lambda S, L, mu: (1 / norm_constant(S)) * torch.exp(exponential(L, mu))
        
        component_likelihoods = [weights[i] * density_function(self.sigmas[i], torch.linalg.cholesky(self.sigmas[i]), self.means[i]) for i in range(self.n_clusters)]

        return torch.stack(component_likelihoods, dim=0).sum(dim=0)
    

    def log_likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(X))
    

    def neglog_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        return - (self.log_likelihoods(X).logsumexp(dim=0) / X.shape[0])
    

    def forward(self, X: torch.Tensor, it: int, validate: bool = False) -> torch.Tensor:
        if validate and (it % 100 == 0): self._validation(X, it)
        out = self.neglog_likelihood(X)

        if not self.monitor: return out
        
        self.add_means(self.means, it)
        self.add_weights(self.tb_params['weights'], it)
        self.add_loss(out, it)

        return out
    

    def val_loss(self, X: torch.Tensor, it: int) -> torch.Tensor:
        out = self.neglog_likelihood(X)
        self.add_valloss(out, it)

        return out
    

    #===================================================================================================
    #                                       VISUALISATION METHODS
    #===================================================================================================

    def plot_contours(self, samples: torch.Tensor, save_to: str):
        _, ax = plt.subplots()
        x, y = samples[:,0], samples[:,1]
        ax.scatter(x, y, s=0.5)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        for i in range(self.n_clusters):
            sigma = self.sigmas[i].data.cpu().numpy()
            mu = self.means[i].data.cpu().numpy()
            
            colour = 'blue' if self.tb_params['weights'][i] > 0 else 'red'
            config = {
                'edgecolor': colour,
                'facecolor': colour,
                'alpha': .2
            }

            self._confidence_ellipse(ax, sigma, mu, **config)
            ax.scatter(mu[0], mu[1], c='red', s=3)
        
        ax.set_title('Monotonic Gaussian Mixture')
        plt.savefig(save_to)


    def plot_heatmap(self, samples: torch.Tensor, save_to: str):
        grid, _, _ = self.create_grid()
        log_likelihoods = self.log_likelihoods(grid)

        heatmap = self._create_heatmap(log_likelihoods)

        levels = np.linspace(-4, 4, 100)
        plt.contour(
            heatmap,
            extent=(self.vmin, self.vmax, self.vmin, self.vmin),
        )

        plt.scatter(samples[:, 0], samples[:, 1], color="k")
        plt.savefig(save_to)