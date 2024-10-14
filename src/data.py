# Classes containing the logic required in the data classes

import torch
import abc

from src.utils.prob_utils import KL_dirichlet_and_uniform_dirichlet, dirichlet_mean, dirichlet_cov, multinomial_cov, multinomial_kl, multinomial_mean, dirichlet_mode

from scipy.signal import savgol_filter

from src.utils.tensor_utils import make_grid


class AbstractObservations(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_inputs(self) -> torch.Tensor:
        # Returns tensor of input locations in M x D  format 
        pass

    @abc.abstractmethod
    def get_observations(self) -> torch.Tensor:
        # Returns tensor of observations in N x M format
        pass
    
    @property 
    def num_data_points(self): 
        return self.get_observations().shape[0]
    
    @property 
    def num_measurements_per_datapoint(self): 
        return  self.get_observations().shape[1]



class SpectralData(AbstractObservations):
    def __init__(self, 
                 wavelengths:torch.Tensor, 
                 spectra: torch.Tensor, 
                 device: torch.device = torch.device('cpu')
                ):
        torch.set_default_dtype(torch.float64)
        self.wavelengths = wavelengths
        self.spectra = spectra
        self.validate_input_shapes()

    def validate_input_shapes(self):
        try:
            assert len(self.wavelengths.shape) == 2
            assert self.wavelengths.shape[1] == 1 
        except AssertionError:
            raise ValueError("Wavelengths should be in num_wavelengths x 1 tensor")
        try:
            assert self.wavelengths.shape[0] == self.spectra.shape[1]
        except AssertionError:
            raise ValueError(
                "spectra should be passed in as a [num samples x num wavelengths] tensor. "
            )

    def mean_center_by_spectra(self):
        self.spectra = self.spectra - self.spectra.mean(axis = 1)[:, None]

    #TODO: Tests for these 
    def mean_center(self):
        self.spectra = self.spectra - self.spectra.mean()

    def scale(self):
        self.spectra = self.spectra/torch.sqrt(self.spectra.var())

    def normalize(self):
        self.mean_center()
        self.scale()

    def snv(self):
        self.mean_center_by_spectra()
        self.spectra = self.spectra/torch.sqrt(self.spectra.var(axis = 1))[:, None]

    def scale_and_center_by_wavelength(self):
        self.spectra = (self.spectra - self.spectra.mean(axis = 0))/torch.sqrt(self.spectra.var(axis=0))

    def trim_wavelengths(self, 
                         min_wavelength = None, max_wavelength = None):
        if min_wavelength is not None:
            mask = min_wavelength <self.wavelengths
            self.wavelengths = self.wavelengths[mask].reshape(-1,1)
            self.spectra = self.spectra[mask.reshape(1,-1).repeat(self.num_data_points,1)].reshape(self.num_data_points, -1)
        if max_wavelength is not None:
            mask = self.wavelengths < max_wavelength
            self.wavelengths = self.wavelengths[mask].reshape(-1,1)
            self.spectra = self.spectra[mask.reshape(1,-1).repeat(self.num_data_points,1)].reshape(self.num_data_points, -1)

    def filter_by_intensity(self,
                            min_intensity = None, max_intensity = None):
        if min_intensity is not None:
            idx = torch.any(self.spectra<min_intensity, dim=1)
            self.spectra = self.spectra[idx]
        if max_intensity is not None:
            idx = torch.any(self.spectra>max_intensity, dim=1)
            self.spectra = self.spectra[idx]

    def zero_min_value(self) -> None:
        self.spectra = self.spectra - self.spectra.min(axis=1).values[:, None]

    def savgol(self, window_size, polynomial_order, derivative):
        self.spectra = torch.Tensor(savgol_filter(self.spectra, window_size,polynomial_order,derivative))

    def to(self, device):
        self.components = self.components.to(device)
        self.spectra = self.spectra.to(device)
        self.wavelengths = self.wavelengths.to(device)

    def get_inputs(self) -> torch.Tensor:
        return self.wavelengths
    
    def get_observations(self) -> torch.Tensor:
        return self.spectra

    @property 
    def num_wavelengths(self): 
        return int(self.wavelengths.shape[0])
    

class Images(AbstractObservations):
    def __init__(self, 
                 images: torch.Tensor 
    )-> None:
        self.images = images.type(torch.float64)
        self.pixel_location = make_grid(torch.linspace(0,1, images.shape[-2]).reshape(-1,1), torch.linspace(0,1, images.shape[-1]).reshape(-1,1))


    def get_inputs(self):
        return self.pixel_location
    
    def get_observations(self):
        return self.images.reshape(self.images.shape[0], -1)
    
    def mean_center(self):
        self.images = self.images - self.images.mean()

    def scale(self):
        self.images = self.images/torch.sqrt(self.images.var())

    def normalize(self):
        self.mean_center()
        self.scale()

class IndependentObservations(AbstractObservations):
    def __init__(self, 
                 observations: torch.Tensor,
                 device: torch.device = torch.device('cpu')
    )-> None:
        self.obs = observations.reshape((observations.shape[0], -1)).type(torch.float64)
        self.inputs = torch.arange(observations.shape[1])
        # self.original_input_shape = make_grid(observations.shape[1])


    def get_inputs(self):
        return self.inputs
    
    def get_observations(self):
        return self.obs
    
    def mean_center(self):
        self.obs = self.obs - self.obs.mean()

    def scale(self):
        self.obs = self.obs/torch.sqrt(self.obs.var())

    def normalize(self):
        self.mean_center()
        self.scale()

    
class AbstractComponents(abc.ABC):
    @abc.abstractmethod
    def get_r(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_r_mode(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_r_outer(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_prior_term(self) -> torch.Tensor:
        ...

    @property 
    def num_data_points(self): 
        return self.get_r().shape[0]

    @property 
    def num_components(self): 
        return self.get_r().shape[1]
    
class ObservedComponents(AbstractComponents):
    def __init__(self, components: torch.tensor, device: torch.device = torch.device('cpu')) -> None:
        torch.set_default_dtype(torch.float64)
        self.components = components
        self.validate_inputs()
        self.device = device
        
    def validate_inputs(self):
        if len(self.components.shape) != 2:
            raise ValueError("Input must be 2D tensor")

    def get_r(self) -> torch.Tensor:
        return self.components
    
    def get_r_mode(self) -> torch.Tensor:
        return self.components
    
    def get_r_outer(self) -> torch.Tensor:
        return self.components[:, None, :] * self.components[:, :, None]
    
    #TODO: tests
    def get_prior_term(self) -> torch.Tensor:
        distribution = torch.distributions.Dirichlet(torch.ones(self.num_components, device=self.device))
        p = torch.zeros(1, device=self.device)
        for r in self.components:
            p -= distribution.log_prob(r)
        return p

class VariationalDirichletDistribution(torch.nn.Module, AbstractComponents):
    def __init__(self, alpha: torch.tensor, device: torch.device = torch.device('cpu')) -> None:
        super(VariationalDirichletDistribution, self).__init__()
        torch.set_default_dtype(torch.float64)
        self._log_alpha = torch.nn.Parameter(torch.log(alpha).contiguous())
        self._log_prior_alpha = torch.zeros(alpha.shape[1], device=device)

    def get_r(self):
        return dirichlet_mean(self.alpha)
    
    def get_r_mode(self):
        return dirichlet_mode(self.alpha)

    def get_r_outer(self):
        mu_r = dirichlet_mean(self.alpha)
        return mu_r[:, None, :]*mu_r[:, :, None] + dirichlet_cov(self.alpha)
    
    def get_prior_term(self) -> torch.Tensor:
        kl = 0
        for a in self.alpha:
            kl += KL_dirichlet_and_uniform_dirichlet(a,self.prior_alpha)
        return -kl

    @property 
    def alpha(self): 
        return  torch.exp(self._log_alpha)
    
    @alpha.setter
    def alpha(self, alpha):
        self._log_alpha = torch.nn.Parameter(torch.log(alpha))

    @property 
    def prior_alpha(self): 
        return  torch.exp(self._log_prior_alpha)

class Deterministic(torch.nn.Module, AbstractComponents):
    def __init__(self, components: torch.tensor) -> None:
        super(Deterministic, self).__init__()
        torch.set_default_dtype(torch.float64)

        self._log_alpha = torch.nn.Parameter(torch.log(components).contiguous())

    def get_r(self):
        alpha = self.alpha
        alpha_0 = alpha.sum(axis=-1)
        return alpha/alpha_0[:, None]

    @property 
    def alpha(self): 
        return torch.exp(self._log_alpha)

    def get_r_outer(self):
        return 
    
    def get_prior_term(self) -> torch.Tensor:
        return 
    
    @alpha.setter
    def alpha(self, alpha):
        self._log_alpha = torch.nn.Parameter(torch.log(alpha))

    
class VariationalClassifier(torch.nn.Module, AbstractComponents):
    def __init__(self, p: torch.Tensor, device: torch.device = torch.device('cpu')) -> None:
        super(VariationalClassifier, self).__init__()
        torch.set_default_dtype(torch.float64)
        assert torch.all(torch.isclose(p.sum(axis=-1),torch.ones_like(p.sum(axis=-1))))
        self._log_p = torch.nn.Parameter(torch.log(p).contiguous())
        self._log_prior_p = torch.zeros(p.shape[1], device=device)
    
    def get_r(self):
        return multinomial_mean(self.p)

    def get_r_mode(self):
        # mode and mean are the same in this case
        return multinomial_mean(self.p)

    def get_r_outer(self):
        mu_r = multinomial_mean(self.p)
        return mu_r[:, None, :]*mu_r[:, :, None] + multinomial_cov(self.p)
    
    def get_prior_term(self):
        return -multinomial_kl(self.p, torch.ones_like(self.p)*1/self.p.shape[1])
    
    @property
    def p(self):
        return torch.exp(self._log_p)/torch.exp(self._log_p).sum(axis = -1).reshape(-1,1)

    @p.setter
    def p_setter(self, p):
        assert torch.all(p.sum(axis=-1)==1)
        self._log_p = torch.nn.Parameter(torch.log(p))
    
    @property 
    def prior_p(self): 
        return  torch.exp(self._log_prior_p)/torch.exp(self._log_prior_p).sum()

    

class Dataset(torch.nn.Module):
    def __init__(self, 
                observations_data: AbstractObservations, 
                components_distribution: AbstractComponents,
                mu_x: torch.Tensor,
                Sigma_x: torch.Tensor,
                device: torch.device = torch.device('cpu')) -> None:
        super(Dataset, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.observations = observations_data
        self.components_distribution = components_distribution

        if mu_x is not None:
            self.mu_x = torch.nn.Parameter(mu_x.contiguous())
            self._log_Sigma_x = torch.nn.Parameter(torch.log(Sigma_x.contiguous()))
        else:
            self.mu_x = None
            self._log_Sigma_x = None

        self.validate_input_shapes()

    def validate_input_shapes(self):
        if self.observations.num_data_points != self.components_distribution.num_data_points:
            raise ValueError(
                    "Number of samples needs to be same for components and samples"
                )
        if self.mu_x is not None:
            if self.observations.num_data_points != self.mu_x.shape[0]:
                raise ValueError(
                        "Number of latent variables needs to match number of observations"
                    )
            if self.observations.num_data_points != self.Sigma_x.shape[0]:
                raise ValueError(
                        "Number of latent variables needs to match number of observations"
                    )
        
    def get_r(self):
        return self.components_distribution.get_r()
    
    def get_r_mode(self):
        return self.components_distribution.get_r_mode()
    
    def get_r_outer(self):
        return self.components_distribution.get_r_outer()

    @property 
    def Sigma_x(self): 
        return  torch.exp(self._log_Sigma_x)
    
    @property 
    def num_data_points(self): 
        return  self.observations.num_data_points
    
    @property 
    def num_measurements_per_data_point(self): 
        return  self.observations.num_measurements_per_datapoint
    
    @property 
    def num_components(self): 
        return  self.components_distribution.num_components
    

        
def get_static_pure_component_spectra(observations_data: AbstractObservations, components_data: ObservedComponents) -> torch.Tensor:
    observations = observations_data.get_observations().type(torch.float32)
    components = components_data.get_r().type_as(observations)
    return (torch.linalg.inv((components.T @ components)) @ (components.T @ observations)).type(torch.float64)

#todo: tests for this 
def get_init_values_for_latent_variables(observations: AbstractObservations, training_observations: AbstractObservations, training_components: ObservedComponents, num_latent_dimensions):
    R_hat = predict_components_from_static_spectra(observations, training_observations, training_components)
    S_hat = get_static_pure_component_spectra(training_observations, training_components)
    E = observations.get_observations() - R_hat@S_hat
    U, _, _ = torch.linalg.svd(E)
    return U[:,:num_latent_dimensions]/torch.sqrt(U[:, :num_latent_dimensions].var(axis = 0))

def predict_components_from_static_spectra(test_observations: AbstractObservations, training_observations: AbstractObservations, training_components: ObservedComponents) -> torch.Tensor:
    S_hat = get_static_pure_component_spectra(training_observations, training_components)
    return  test_observations.get_observations() @ torch.linalg.pinv(S_hat)
