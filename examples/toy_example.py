# Toy example from the paper

# %%
import copy
from src.wsgplvm import WSGPLVM
from src.data import Dataset, ObservedComponents, SpectralData, VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.tensor_utils import  log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data

from matplotlib.lines import Line2D
import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import cauchy
from itertools import chain


import pandas as pd


# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234567890)
np.random.seed(1234567890)


# %%
# Define the training set parameters
num_training_samples = 100
wavelengths = torch.linspace(-15, 15, 101)
sigma2 = torch.Tensor([1e-2])
# %%
# Define model parameters

def get_pure_components(latent_variable):
    centroids = torch.vstack((latent_variable*2 - 7.5,  latent_variable*2 + 7.5))
    amplitudes = torch.vstack((2*latent_variable + 15, 2 * latent_variable + 10))
    return pure_component(centroids, amplitudes).permute(2,1,0)

def pure_component(mu, a, wavelengths=wavelengths):
    return a[:, None, : ] * cauchy.pdf((wavelengths[None, :, None] - mu[:, None, :]) / 3)

def get_observations(components, pure_component_spectra):
    noise_free_observations = (components[:, None, :] * pure_component_spectra[:, :, :]).sum(axis=-1) 
    return noise_free_observations + torch.randn_like(noise_free_observations)*torch.sqrt(sigma2)


choices = [[0.2,0.8]]*int(num_training_samples/4) + [[0.4,0.6]]*int(num_training_samples/4) +  [[0.6,0.4]]*int(num_training_samples/4) + [[0.8,0.2]]*int(num_training_samples/4)

training_components = torch.Tensor(choices)
true_latent_points = torch.randn(num_training_samples)

training_pure_component_spectra = get_pure_components(true_latent_points)

training_observed_spectra = get_observations(training_components, training_pure_component_spectra)

# %%

training_spectra = SpectralData(wavelengths.reshape(-1,1), training_observed_spectra)
training_components = ObservedComponents(training_components)
# %%

num_latent_dims = 5
init_latent_point = get_init_values_for_latent_variables(training_spectra, training_spectra, training_components, num_latent_dims)

training_dataset = Dataset(training_spectra, training_components, copy.deepcopy(init_latent_point), torch.ones_like(init_latent_point))

loss = []
# %%

wsgplvm = WSGPLVM(
    beta = torch.ones(init_latent_point.shape[1]), 
    gamma=torch.ones(1)*10, 
    sigma2 = torch.ones(1)*1, 
    sigma2_s = torch.ones(1)*1,
    v_x = torch.randn(8,num_latent_dims),
    v_l= torch.linspace(-10,10,16).reshape(-1,1)
)

# %%
adam_training = torch.optim.Adam(chain(wsgplvm.parameters(), training_dataset.parameters()), lr = 1e-2)

loss.extend(train_bass_on_spectral_data(wsgplvm, [training_dataset], adam_training, 2000))



from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(true_latent_points.reshape(-1,1), training_dataset.mu_x.detach().numpy())
true_vals = [[0],[-2],[2]]
preds = lm.predict(np.array(true_vals))

with torch.no_grad():
    for i in range(3):
        x = torch.Tensor(preds)
        mu_l = wsgplvm.get_point_mean(x[i].reshape(1,-1), [training_dataset])
        sigma_l = wsgplvm.get_point_var(x[i].reshape(1,-1), [training_dataset])
        ground_truth = get_pure_components(torch.Tensor(true_vals[i]))
        plt.figure()
        plt.plot(wavelengths, mu_l[0])
        plt.plot(wavelengths, mu_l[1])
        plt.plot(wavelengths, ground_truth[0,:,0])
        plt.plot(wavelengths, ground_truth[0,:,1])
        plt.fill_between(wavelengths, mu_l[0]+2*torch.sqrt(sigma_l[0]), mu_l[0]-2*torch.sqrt(sigma_l[0]), alpha = 0.5)
        plt.fill_between(wavelengths, mu_l[1]+2*torch.sqrt(sigma_l[1]), mu_l[1]-2*torch.sqrt(sigma_l[1]), alpha = 0.5)
        plt.show()

# %%
        
noise_schedule = log_linspace(1,wsgplvm.sigma2.item(), 20)





with torch.no_grad():

    s = wsgplvm.get_sample_mean(training_dataset, [training_dataset])
    for i in range(1):
        fig, ax = plt.subplots()
        # plt.title(f"reconstruction {i}")
        ax.plot(wavelengths, s[i, :,0], label = "reconstructed spectra 1")
        ax.plot(wavelengths, s[i, :,1], label = "reconstructed spectra 2")
        plt.plot(wavelengths, training_pure_component_spectra[i, :,0], label = "real spectra 1")
        ax.plot(wavelengths, training_pure_component_spectra[i, :,1], label = "real spectra 2")
        ax.legend()
        ax.set_xlabel("Wavelengths")
    # plt.savefig("examples/figs/toy_example/reconstruction.pdf", bbox_inches='tight')
plt.show()




# %%

print("beta", wsgplvm.beta)
print("gamma", wsgplvm.gamma)
print("sigma2", wsgplvm.sigma2)
print("sigma2_s", wsgplvm.sigma2_s)
# %%

num_test_samples = 100

test_components = torch.distributions.Dirichlet(torch.ones(2)).sample((num_test_samples,))
test_true_latent_points = torch.randn(num_test_samples) 


test_pure_component_spectra = get_pure_components(test_true_latent_points)
test_observed_spectra = get_observations(test_components, test_pure_component_spectra)


test_spectra = SpectralData(wavelengths.reshape(-1,1), test_observed_spectra)
# test_spectra = IndependentObservations(test_observed_spectra)
# %%


estimated_components = VariationalDirichletDistribution(torch.ones(test_spectra.num_data_points, 2))
# estimated_components = VariationalDirichletDistribution(torch.ones(num_training_samples, 2))
test_dataset = Dataset(test_spectra, estimated_components, torch.zeros(num_test_samples, num_latent_dims), Sigma_x=2*torch.ones(num_test_samples, num_latent_dims))


test_loss = []
# 
# %%


for s in noise_schedule:
    print(s)
    wsgplvm.sigma2 = s 
    test_loss.extend(lbfgs_training_loop(wsgplvm, [training_dataset, test_dataset], test_dataset.parameters(), 2))

test_loss.extend(lbfgs_training_loop(wsgplvm, 
                                     [training_dataset, test_dataset], 
                                     chain(test_dataset.parameters(), training_dataset.parameters(), wsgplvm.parameters()),
                                     20))


# %%
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(4)
pls.fit(training_dataset.observations.get_observations(), training_dataset.components_distribution.get_r())
y_hat = pls.predict(test_dataset.observations.get_observations())



# %%
# Set the font to be LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 26  # Adjust the font size as needed
})

with torch.no_grad():
    vars = torch.Tensor([])
    for a in test_dataset.components_distribution.alpha:
        dist = torch.distributions.Dirichlet(a)
        vars = torch.concat((vars,dist.variance))
    vars = vars.reshape(-1,2)
    fig, ax = plt.subplots(2, figsize = (8,12))
    arg_max = torch.argmax(test_dataset.mu_x[:,-1])
    ax[0].errorbar(test_components[:,0], test_dataset.get_r()[:, 0], 2*torch.sqrt(vars[:,0]), linestyle="None", capsize=2, marker= ".", alpha = 1, ) 
    # plt.scatter(test_components[:,i], y_hat[:,i], c = "orange")
    ax[0].plot(test_components[:,0], test_components[:,0])
    ax[0].set_xlabel(f"True Mixture Component 1")
    ax[0].set_ylabel(f"Estimated Mixture Component 1")
    relevant_lv = torch.argmin(wsgplvm.beta)
    ax[1].errorbar(true_latent_points.flatten(), training_dataset.mu_x[:, relevant_lv], 2*torch.sqrt(training_dataset.Sigma_x[:, relevant_lv]), linestyle="None", capsize=2, marker= ".", alpha = 1, label = "Training Data")
    ax[1].errorbar(test_true_latent_points.flatten(), test_dataset.mu_x[:, relevant_lv], 2*torch.sqrt(test_dataset.Sigma_x[:, relevant_lv]), linestyle="None", capsize=2, marker= ".", alpha = 1, label = "Test Data")
    ax[1].legend()
    ax[1].set_xlabel("True Latent Variable")
    ax[1].set_ylabel("Estimated Latent Variable")
    plt.show()



# %%
for i in range(50):
    beta = torch.distributions.Beta(test_dataset.components_distribution.alpha[i,0],
                            test_dataset.components_distribution.alpha[i,:].sum())

    lins = torch.linspace(0,1,100)
    with torch.no_grad():
        plt.plot(lins, beta.log_prob(lins).exp())

# %%
from matplotlib.lines import Line2D



cmap = plt.get_cmap("plasma")
with torch.no_grad():
    fig, ax = plt.subplots(2,figsize = (8,12))
    for i in range(training_spectra.num_data_points):
        ax[0].plot(training_spectra.spectra[i], c = cmap(training_components.get_r()[i,0]))
    legend_elements = [
    Line2D([0], [0], color=cmap(0.2),  label='$r_1 = 0.2$, $r_2 = 0.8$'),
    Line2D([0], [0], color=cmap(0.4),  label='$r_1 = 0.4$, $r_2 = 0.6$'),
    Line2D([0], [0], color=cmap(0.6),  label='$r_1 = 0.6$, $r_2 = 0.4$'),
    Line2D([0], [0], color=cmap(0.8),  label='$r_1 = 0.8$, $r_2 = 0.2$'),
    ]
    ax[0].legend(handles = legend_elements)



    
    for i in range(test_spectra.num_data_points):
        ax[1].plot(test_spectra.spectra[i], c = "grey", alpha = 0.5)
    legend_elements = [
    Line2D([0], [0], color="grey",  label= "Unknown component fractions"), 
    ]
    ax[1].legend(handles = legend_elements)
    ax[0].set_ylabel("$y$")
    ax[1].set_ylabel("$y^*$")
    ax[1].set_xlabel("$\lambda$")
    fig.subplots_adjust(hspace = 0.1)
    plt.show()

# %%

from sklearn.linear_model import LinearRegression
import matplotlib.lines as mlines


lm = LinearRegression().fit(true_latent_points.reshape(-1,1), training_dataset.mu_x.detach().numpy())
true_vals = [[-2],[0],[2]]
preds = lm.predict(np.array(true_vals))

cmap = plt.cm.tab10
from matplotlib.patches import Patch
with torch.no_grad():
    fig, ax = plt.subplots(2,figsize = (8,12))
    for i in range(3):
        for j in range(2):
            x = torch.Tensor(preds)
            mu_l = wsgplvm.get_point_mean(x[i].reshape(1,-1), [training_dataset])
            sigma_l = wsgplvm.get_point_var(x[i].reshape(1,-1), [training_dataset])
            ground_truth = get_pure_components(torch.Tensor(true_vals[i]))
            
            ax[j].plot(wavelengths, mu_l[j], c = cmap(i/10))
            ax[j].plot(wavelengths, ground_truth[0,:,j],c = cmap(i/10), ls = "--")
            ax[j].fill_between(wavelengths, mu_l[j]+1.96*torch.sqrt(sigma_l[j]), mu_l[j]-1.96*torch.sqrt(sigma_l[j]), color = cmap(i/10), alpha = 0.5)
    true_function_handle = mlines.Line2D([], [], color='black', linestyle='--', label='True')
    mean_estimated_handle = mlines.Line2D([], [], color='black', linestyle='-', label='GP Mean')
    ci_handle = Patch(color='black', alpha=0.2, label='GP Confidence Interval')

    # Handles for different x values with specific colors
    x_minus_2_handle = mlines.Line2D([], [], color='blue', label='h = -2')
    x_zero_handle = mlines.Line2D([], [], color='orange', label='h = 0')
    x_two_handle = mlines.Line2D([], [], color='green', label='h = 2')
    ax[0].legend(handles=[true_function_handle, (mean_estimated_handle, ci_handle), x_minus_2_handle, x_zero_handle, x_two_handle], labels = ["True Signals", "GP Estimate", "h = -2", "h = 0", "h=2"])
    ax[1].set_xlabel("$\lambda$")
    ax[0].set_ylabel("$f_1(h, \lambda)$")
    ax[1].set_ylabel("$f_2(h, \lambda)$")
    fig.subplots_adjust(hspace = 0.1)
    plt.show()

# %%
