import torch
import numpy as np 

def diff_between_rank_2_tensors(x, y):
    try:
        assert len(x.shape) == len(y.shape) == 2
        assert x.shape[-1] == y.shape[-1]
    except AssertionError:
        raise ValueError(
            f"diff_between_rank_2_tensors got invalid argument. This should only be used on a NxA and MxA matrix to make a NxMxA tensor. Got tensors of shape {x.shape} and {y.shape}"
        )
    return x[:, None, :] - y[None, :, :]


def L2_norm_along_final_dim(x: torch.Tensor, dim_scales: torch.Tensor =None):
    if dim_scales is None:
        return (x**2).sum(axis=-1).squeeze()
    try:
        assert all(dim_scales >= 0)
    except AssertionError:
        print(dim_scales)
        raise ValueError("dim scales must be positive")
    try:
        assert dim_scales.shape[0] == x.shape[-1]
    except AssertionError:
        raise ValueError("number of scales must match final dimension of tensor")
    return (x**2 * dim_scales).sum(axis=-1).squeeze()

def weighted_L2_diff(x, y, dim_scales=None):
    diff = diff_between_rank_2_tensors(x, y)
    return L2_norm_along_final_dim(diff, dim_scales)


def invert_covariance_matrix(Sigma):
    
    Sigma_chol = torch.linalg.cholesky(Sigma)

    return torch.cholesky_inverse(Sigma_chol)

def tile_final_dims_to_matrix(
        tensor: torch.Tensor,
    ):
        dims = []
        if len(tensor.shape) > 4: 
            dims = list(tensor.shape[:-4])
        dims.append(tensor.shape[-4]*tensor.shape[-2])
        dims.append(tensor.shape[-3]*tensor.shape[-1])
        return tensor.swapaxes(-3,-2).reshape(dims)

def get_pca_latent_values(
        E: torch.Tensor, 
        num_dims: int
    ):
    E = E - E.mean(axis=0)
    U, S, Vh = torch.linalg.svd(E)
    return (U[:, :num_dims]) / torch.sqrt(U[:, :num_dims].var(axis=0))

def make_grid(a: torch.Tensor,b: torch.Tensor) -> torch.Tensor:
    a_rep = a.repeat_interleave(b.shape[0],dim = 0)
    b_rep = b.repeat(a.shape[0],1)
    return torch.hstack((a_rep,b_rep))





def log_linspace(start, end, num_steps = 5):
    return np.exp(np.linspace(np.log(start),np.log(end),num_steps))


def invert_kronecker_matrix(B:torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Inverts a block diagonal matrix
    
    Using the identity that (A kron B)^-1 = A^-1 kron B^-1 
    """

    K_inv = invert_covariance_matrix(K)

    B_inv = invert_covariance_matrix(B)

    return  torch.kron(B_inv, K_inv)
