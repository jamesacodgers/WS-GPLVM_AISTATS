import torch 

def kl_between_mean_field_norm_and_standard_norm(
        means: torch.tensor,
        vars: torch.tensor, 
):
    expanded =  1/2* (means**2 + vars - 1 -  torch.log(vars))
    return expanded.sum()
     
def dirichlet_mean(
        alpha: torch.tensor, 
) -> torch.tensor:
    alpha_0 = alpha.sum(axis=-1)
    return alpha/alpha_0[:, None]

def dirichlet_mode(
        alpha: torch.tensor, 
) -> torch.tensor:
    alpha_0 = alpha.sum(axis=-1)
    K = alpha.shape[-1]

    if torch.isnan((alpha-1)/(alpha_0[:, None] - K)).any():
        print('nans detected in mode')
        
    return (alpha-1)/(alpha_0[:, None] - K)

def dirichlet_cov(
       alpha: torch.tensor, 
) -> torch.Tensor:
    alpha_tilde = dirichlet_mean(alpha)
    outer_alpha = alpha_tilde[:,None, :]*alpha_tilde[:,:,None]
    diag_alpha = torch.diag_embed(alpha_tilde)
    return (diag_alpha - outer_alpha)/(alpha.sum(axis=-1)[:, None, None] + 1)

def KL_dirichlet_and_uniform_dirichlet(
        alpha: torch.Tensor,
        prior_alpha: torch.Tensor
) -> torch.Tensor:
    prior = torch.distributions.Dirichlet(prior_alpha*torch.ones_like(alpha))
    variational_dist = torch.distributions.Dirichlet(alpha)
    return torch.distributions.kl_divergence(variational_dist, prior)
#TODO: test
def multinomial_mean(p: torch.Tensor) -> torch.Tensor:
    return p
#TODO: test
def multinomial_cov(p: torch.Tensor) -> torch.Tensor:
    cov = - p[:, None, :]* p[:, :, None]
    var_diag = torch.diag_embed(p)
    return cov + var_diag
#TODO: test
def multinomial_kl(p_1: torch.Tensor, p_2:torch.Tensor) -> torch.Tensor:
    return (torch.log(p_1/p_2)*p_1).sum()