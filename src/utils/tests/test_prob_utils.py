import torch

from src.utils.prob_utils import KL_dirichlet_and_uniform_dirichlet, dirichlet_cov, dirichlet_mean, kl_between_mean_field_norm_and_standard_norm 

def test_kl_from_mean_field_to_standard_normal():
    mu_tensor = torch.randn(100,10)
    sigma_tensor = torch.randn(100,10)**2
    prior = torch.distributions.MultivariateNormal(torch.zeros(10), torch.eye(10))
    KL = 0
    for mu, sigma in zip(mu_tensor, sigma_tensor):
        dist = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        KL += torch.distributions.kl.kl_divergence(dist, prior)
    result = kl_between_mean_field_norm_and_standard_norm(mu_tensor,sigma_tensor)
    assert torch.isclose(KL, result)

def test_dirichlet_mean():
    alpha = torch.rand(100,5)*10
    expected_means = torch.zeros_like(alpha)
    for i,a in enumerate(alpha):
        dist = torch.distributions.Dirichlet(a)
        expected_means[i] = dist.mean
    means = dirichlet_mean(alpha)
    assert torch.all(means == expected_means)

def test_dirichlet_cov():
    alpha = torch.rand(100,5)*10
    expected_cov = torch.zeros(100,5,5)
    for i,a in enumerate(alpha):
        alpha_0 = a.sum()
        a_tilde = a/alpha_0
        for j,a1 in enumerate(a_tilde):
            for k,a2 in enumerate(a_tilde):
                expected_cov[i,j,k] =  - a1*a2/(alpha_0+1)
                if j == k:
                    expected_cov[i,j,k] +=  a1/(alpha_0+1)
    cov = dirichlet_cov(alpha)
    assert torch.all(torch.isclose(cov, expected_cov))

def test_KL_dirichlet():
    torch.manual_seed(1234)
    alphas = torch.rand(100,3)*5
    expected_kls = torch.zeros(100)
    prior = torch.distributions.Dirichlet(torch.ones(3))
    for i in range(100):
        dist = torch.distributions.Dirichlet(alphas[i])
        expected_kls[i] = torch.distributions.kl.kl_divergence(dist,prior)
    kls = KL_dirichlet_and_uniform_dirichlet(alphas, torch.ones(3))
    assert torch.all(kls == expected_kls)