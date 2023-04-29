# External imports
import torch
import numpy as np

# Local imports
from .initialisation_procedures import GMMInitalisation, check_random_state

def create_nm_initialisation(
        components: int,
        init_type: str,
        covar_shape: str,
        covar_reg: float,
        train_set: torch.tensor,
        optimal_nm_shape: str
):
    initaliser_nmgmm = GMMInitalisation(
        n_components=components,
        init_params=init_type,
        covariance_type=covar_shape,
        reg_covar=covar_reg
    )
    initaliser_gmm = GMMInitalisation(
        n_components=components ** 2,
        init_params=init_type,
        covariance_type=covar_shape,
        reg_covar=covar_reg
    )
    random_seed = check_random_state(None)

    initaliser_nmgmm.initialize_parameters(train_set, random_seed)
    initaliser_gmm.initialize_parameters(train_set, random_seed)

    _covariances_nmgmm = initaliser_nmgmm.covariances_
    _covariances_gmm = initaliser_gmm.covariances_

    _means_nmgmm = initaliser_nmgmm.means_
    _means_gmm = initaliser_gmm.means_

    _weights_nmgmm = None

    if covar_shape == 'diag':
        _covariances_nmgmm = np.array([np.diag(np.sqrt(S)) for S in _covariances_nmgmm])
        _covariances_gmm = np.array([np.diag(S) for S in _covariances_gmm])


    if optimal_nm_shape == 'funnel' and components == 3:
        _means_nmgmm[0] = [3.5, 4] 
        _means_nmgmm[1] = [3.5, -4]
        _means_nmgmm[2] = [-1, 0] 

        _covariances_nmgmm[0] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[1] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[2] = [[7, 0], [0, 7]]

        _weights_nmgmm = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)

    if optimal_nm_shape == 'funnel' and components == 5:
        _means_nmgmm[0] = [3.5, 4] 
        _means_nmgmm[1] = [3.5, -4]
        _means_nmgmm[2] = [3.5, 4] 
        _means_nmgmm[3] = [3.5, -4]
        _means_nmgmm[4] = [-1, 0] 

        _covariances_nmgmm[0] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[1] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[2] = [[2, 0], [-1, 1.5]]
        _covariances_nmgmm[3] = [[2, 0], [1, 1.5]]
        _covariances_nmgmm[4] = [[6, 0], [0, 6]]

        _weights_nmgmm = torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001], dtype=torch.float64)

    if optimal_nm_shape == 'mor' and components == 3:
        init_zip = zip(
            [torch.tensor([0, 0]), torch.tensor([1.5, 1.5]), torch.tensor([0.5, 0.5])], 
            torch.torch.from_numpy(initaliser_nmgmm.covariances_)
        )
        _covariances_nmgmm = torch.stack([torch.sqrt(torch.diag(x)) - torch.diag(i) for i, x in init_zip])
        _covariances_nmgmm = _covariances_nmgmm.cpu().numpy()

    if optimal_nm_shape == 'banana' and components == 3:
        _means_nmgmm[0] = [0, 5]
        _means_nmgmm[1] = [0, 10]
        _means_nmgmm[2] = [0, 10]

        _covariances_nmgmm[0] = [[7, 0], [0, 7]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[2] = [[2.5, 0], [0, 5]]

        _weights_nmgmm = torch.tensor([.001, .001, .001])

    if optimal_nm_shape == 'banana' and components == 4:
        _means_nmgmm[0] = [0, 5]
        _means_nmgmm[1] = [0, 10]
        _means_nmgmm[2] = [0, 10]
        _means_nmgmm[3] = [0, 5]


        _covariances_nmgmm[0] = [[7, 0], [0, 7]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[2] = [[2.5, 0], [0, 5]]
        _covariances_nmgmm[3] = [[7, 0], [0, 7]]


        _weights_nmgmm = torch.tensor([.001, .001, .001, .001])

    if optimal_nm_shape == 'cosine' and components == 6:
        _means_nmgmm[0] = [0, 0.5] 
        _means_nmgmm[1] = [-1.5, 3]
        _means_nmgmm[2] = [-0.1, -3]
        _means_nmgmm[3] = [1.5, 3]
        _means_nmgmm[4] = [-3.1, -3]
        _means_nmgmm[5] = [3.1, -3] 

        _covariances_nmgmm[0] = 1.5 * np.array([[3, 0], [0, 3]])
        _covariances_nmgmm[1] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[2] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[3] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[4] = [[0.3, 0], [0, 1.5]]
        _covariances_nmgmm[5] = [[0.3, 0], [0, 1.5]]

        _weights_nmgmm = torch.tensor([.001, .001, .001, .001, .001, .001])

    if optimal_nm_shape == 'mor' and components == 6:
        _means_nmgmm[0] = [0, 0] 
        _means_nmgmm[1] = [0, 0]
        _means_nmgmm[2] = [0, 0]
        _means_nmgmm[3] = [0, 0]
        _means_nmgmm[4] = [0, 0]
        _means_nmgmm[5] = [0, 0]
        _means_nmgmm[6] = [0, 0]
        _means_nmgmm[7] = [0, 0]

        _covariances_nmgmm[0] = [[3.5, 0], [0, 3.5]]
        _covariances_nmgmm[1] = [[2.5, 0], [0, 2.5]]
        _covariances_nmgmm[2] = [[1.5, 0], [0, 1.5]]
        _covariances_nmgmm[3] = [[0.5, 0], [0, 0.5]]
        _covariances_nmgmm[4] = [[2.5, 0], [0, 2.5]]
        _covariances_nmgmm[5] = [[0.5, 0], [0, 0.5]]
        _covariances_nmgmm[6] = [[0.2, 0], [0, 0.2]]
        _covariances_nmgmm[7] = [[0.1, 0], [0, 0.1]]


    if optimal_nm_shape == 'spiral' and components == 4:
        _means_nmgmm[0] = [0, 0] 
        _means_nmgmm[1] = [1.5, 4]
        _means_nmgmm[2] = [1.5, -2.8]
        _means_nmgmm[3] = [-2.5, 0.7]

        _covariances_nmgmm[0] = [[4, 0], [0, 4]]
        _covariances_nmgmm[1] = [[0.3, 1], [-1, 0.8]]
        _covariances_nmgmm[2] = [[1, 1], [-0.5, 0.3]]
        _covariances_nmgmm[3] = [[1, 0.5], [0.5, 0.3]]


        _weights_nmgmm = torch.tensor([0.001, 0.001, 0.001, 0.001])
    

    return [(_means_nmgmm, _covariances_nmgmm, _weights_nmgmm), (_means_gmm, _covariances_gmm)]