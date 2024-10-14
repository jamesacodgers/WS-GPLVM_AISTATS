import torch 
from src.data import Dataset

class CLS(torch.nn.Module):
    def __init__(self,M, C, sigma2=1e-2, gamma=None, sigma2_s=1, wavelengths = None):
        super(CLS, self).__init__()
        self.M = M 
        self.C = C
        self._log_sigma2_s = torch.nn.Parameter(torch.log(torch.tensor(sigma2_s)))

        if gamma!=None: 
            self._log_gamma = torch.nn.Parameter(torch.log(torch.tensor(gamma)))
            assert wavelengths is not None
            self.wavelengths = wavelengths
            diff = self.wavelengths[:,None] - self.wavelengths[None,:]
            K = sigma2_s*torch.exp(-1/(2*self.gamma)*diff**2)
            block = torch.block_diag(*[K]*self.C)
            cov =   block + torch.eye(self.C*self.M)*1e-5
            torch.nn.Parameter(torch.cholesky(cov))
            self._raw_chol_S= torch.nn.Parameter(torch.cholesky(cov))
        else:
            self._log_gamma=None
            self._raw_chol_S= torch.nn.Parameter(sigma2_s*torch.eye(self.C).unsqueeze(0).repeat(self.M,1,1))
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))

        self._mu_S = torch.nn.Parameter(torch.zeros(M*self.C))


    def forward(self):
        ...

    def elbo(self, 
             datasets: list[Dataset]
        ):
        r_prior_term = torch.zeros(1)
        s_independent_terms = torch.zeros(1)
        fit = torch.zeros(1)
        for dataset in datasets: 
            if dataset.components_distribution.num_components != self.C:
                raise ValueError(
                    "There must be the same number of components for each dimension"
                )
            fit += 1/self.sigma2*(dataset.observations.get_observations()*(dataset.get_r()@self.mu_S.T )).sum() - 1/(2*self.sigma2)*torch.trace( (dataset.get_r_outer()@(self.mu_S.T @ self.mu_S + self.Sigma_S.sum(dim=0))).sum(dim=0))
            r_prior_term += dataset.components_distribution.get_prior_term()
            s_independent_terms += (
                -dataset.observations.num_data_points * dataset.observations.num_measurements_per_datapoint/ 2 * torch.log(2 * torch.pi*self.sigma2)
                - 1/ (2 * self.sigma2) * (dataset.observations.get_observations() ** 2).sum()
            )

        kl = self.get_kl_S()
        return (
                fit 
                + r_prior_term
                + s_independent_terms
                - kl
                )
    
    def get_kl_S(self):
        if self._log_gamma:
            variational_dist = torch.distributions.MultivariateNormal(self._mu_S, self._Sigma_S)
            prior = torch.distributions.MultivariateNormal(torch.zeros(self.M*self.C), self.Sigma_prior)
        else:
            variational_dist = torch.distributions.MultivariateNormal(self.mu_S, self.Sigma_S)
            prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.sigma2_s*torch.eye(self.C))
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    @property 
    def gamma(self): 
        return  torch.exp(self._log_gamma)

    @property 
    def sigma2(self): 
        return  torch.exp(self._log_sigma2)
    @property 
    def mu_S(self): 
        return self._mu_S.reshape(self.C, self.M).T
        
    @property 
    def _Sigma_S(self): 
        L = torch.tril(self._raw_chol_S)
        return L.transpose(-2,-1)@L + torch.eye(L.shape[-1])*1e-5
    @property 
    def Sigma_S(self): 
        if self._log_gamma:
            return self._Sigma_S.reshape(self.C,self.M,self.C,self.M).transpose(1,2)[:,:,range(self.M),range(self.M)].permute(2,0,1)
        else: 
            return self._Sigma_S
    @property 
    def sigma2_s(self):
        return torch.exp(self._log_sigma2_s)
    @property 
    def Sigma_prior(self): 
        if self._log_gamma is None:
            return self.sigma2_s*torch.eye(self.C)
        else:
            diff = self.wavelengths[:,None] - self.wavelengths[None,:]
            K = self.sigma2_s*torch.exp(-1/(2*self.gamma)*diff**2)
            block = torch.block_diag(*[K]*self.C)
            return  block + torch.eye(self.C*self.M)*1e-5