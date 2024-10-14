# Near Infra red example from the paper

# %%
import copy
import numpy as np
from scipy.io import loadmat
import torch
from src.mogplvm import MOGPLVM
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.model_selection import train_test_split

from src.data import Dataset, ObservedComponents, SpectralData, VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.tensor_utils import log_linspace 
from src.utils.train_utils import train_bass_on_spectral_data, lbfgs_training_loop
from src.utils.prob_utils import dirichlet_cov, dirichlet_mean
from src.utils.plot_utils import project_vect_to_dirichlet, project_covar_to_dirichlet

from sklearn.cross_decomposition import PLSRegression

torch.set_default_dtype(torch.float64)

torch.manual_seed(1234)
np.random.seed(1234)

data = loadmat("examples/data/flodat.mat")
del(data["SDisk"])

# %%

for key, item in data.items():
    data[str(key)] = torch.Tensor(item)

all_spectra_tensor = torch.hstack([data["spec30"],
    data["spec40"],
    data["spec50"],
    data["spec60"],
    data["spec70"],
]).T

all_spectra_dataset = SpectralData(data["wl"], all_spectra_tensor)
all_spectra_dataset.trim_wavelengths(800,1000)
all_spectra_dataset.snv()


all_spectra_dataset.scale()
remove_frac_of_points = 1
all_spectra, data["wl"] = all_spectra_dataset.spectra[:,::remove_frac_of_points], all_spectra_dataset.wavelengths[::remove_frac_of_points]



all_temperatures = torch.vstack([
    data["temper30"],
    data["temper40"],
    data["temper50"],
    data["temper60"],
    data["temper70"],
])

all_components = data["conc"].repeat(5,1)




training_spectra_tensor, test_spectra_tensor, training_components_tensor, test_components_tensor, training_temp_tensor, test_temp_tensor = train_test_split(all_spectra, all_components, all_temperatures, test_size=0.5)

training_spectra = SpectralData(data["wl"], training_spectra_tensor)



training_components = ObservedComponents(training_components_tensor)

# %%
print(training_components.components)

#%%

num_latent_dims = 5
x_init_train = get_init_values_for_latent_variables(training_spectra, training_spectra, training_components, num_latent_dims)



training_data = Dataset(
                training_spectra, 
                training_components,
                mu_x = copy.deepcopy(x_init_train), 
                Sigma_x = torch.ones_like(x_init_train)*1
                )

test_spectra = SpectralData(data["wl"], test_spectra_tensor)

x_test_init = get_init_values_for_latent_variables(test_spectra, training_spectra, training_components, num_latent_dims)

test_components = VariationalDirichletDistribution(
torch.ones(test_spectra.num_data_points, 3)

)


test_data = Dataset(
                test_spectra, 
                test_components, 
                torch.zeros(test_spectra.num_data_points, num_latent_dims),
                Sigma_x = 1*torch.ones(test_spectra.num_data_points, num_latent_dims
                ))


bass = MOGPLVM(
    beta = torch.ones(x_init_train.shape[1])*5, 
    gamma=torch.ones(1)*1000, 
    sigma2 = torch.ones(1)*1e-0, 
    sigma2_s = torch.ones(1)*1,
    v_x = torch.randn(6,num_latent_dims)*1,
    v_l= torch.linspace(data["wl"].min(),data["wl"].max(),28).reshape(-1,1),
)

loss = []

adam_bass = torch.optim.Adam(chain(bass.parameters(), training_data.parameters()), 5e-2)
adam_train = torch.optim.Adam(chain(bass.parameters(), training_data.parameters()), 5e-2)
adam_r = torch.optim.Adam(chain(test_data.components_distribution.parameters()), 5e-2)
adam_test = torch.optim.Adam(chain(training_data.parameters()), 5e-2)
adam_all = torch.optim.Adam(chain(bass.parameters(), training_data.parameters(),training_data.parameters()), 5e-2)

loss.extend(train_bass_on_spectral_data(bass, [training_data], adam_bass, 1000))

# %%


test_loss = []

sigma_2 = copy.deepcopy(bass.sigma2.item())

noise_schedule = log_linspace(1, sigma_2, 20)
for i in range(1):
    for s in noise_schedule:
        print(s)
        bass.sigma2 = s
        test_loss.extend(lbfgs_training_loop(bass, [training_data, test_data],  chain(test_data.parameters(), training_data.parameters(),), 2))


#%%

test_loss.extend(lbfgs_training_loop(bass, [training_data, test_data], chain(test_data.parameters(), training_data.parameters(), bass.parameters()), 2)) 

while test_loss[-1] - test_loss[-2] > 1 :
    test_loss.extend(lbfgs_training_loop(bass, [training_data, test_data], chain(test_data.parameters(), training_data.parameters(), bass.parameters()), 1)) 


with torch.no_grad():
    plt.plot(test_loss)
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 28  # Adjust the font size as needed
})

nA = 10
msep = torch.zeros(nA,3)

d1,d2, r1, r2 =  train_test_split(training_spectra.get_observations(), training_components.get_r())

for i in range(nA):
    pls_reg = PLSRegression(i+1)
    pls_reg.fit(d1, r1)
    # y_hat = pls_reg.predict(test_spectra.spectra)
    y_hat = pls_reg.predict(d2)
    # plt.figure()
    for j in range(3):
        # msep[i,j] = ((test_components_tensor[:,j] - y_hat[:,j])**2).sum()
        msep[i,j] = ((r2[:,j] - y_hat[:,j])**2).sum()

min_err_components = torch.argmin(msep.sum(axis=1))

pls_reg = PLSRegression(min_err_components.item()+1)
pls_reg.fit(training_spectra.get_observations(), training_components.get_r())
y_hat = pls_reg.predict(test_spectra.get_observations())



plt.figure()
plt.plot(torch.log(msep[:,0]))
plt.plot(torch.log(msep[:,1]))
plt.plot(torch.log(msep[:,2]))
plt.show()
# %%

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(training_spectra.get_inputs(), training_spectra.get_observations()[0], c = "orange", alpha = 0.25, label = "training spectra")
ax.plot(training_spectra.get_inputs(), training_spectra.get_observations()[1:].T, c = "orange", alpha = 0.25)
ax.plot(test_spectra.get_inputs(), test_spectra.get_observations()[0], c = "blue", alpha = 0.25, label = "test spectra")
ax.plot(test_spectra.get_inputs(), test_spectra.get_observations()[1:].T, c = "blue", alpha = 0.25)
ax.set_xlabel("wavelength")
ax.set_ylabel("d")
ax.legend()
fig.savefig("examples/figs/varying_temperature_Wulfert/spectra.pdf", bbox_inches = "tight")
fig.show()
# %%
component_names = ["Ethanol", "Water", "2-Propanol"]

# %%
with torch.no_grad():
    vars = torch.Tensor([])
    for a in test_data.components_distribution.alpha:
        dist = torch.distributions.Dirichlet(a)
        vars = torch.concat((vars,dist.variance))
    vars = vars.reshape(-1,3)
    
    fig, axs = plt.subplots(3, figsize = (6,18))
    for i in range(3):
        axs[i].errorbar(test_components_tensor[:,i], test_data.get_r()[:, i], 2*torch.sqrt(vars[:,i]), linestyle="None", capsize=2, marker= ".", alpha = 1, label = "BAMM prediction", markersize = 10) 
        axs[i].plot(test_components_tensor[:,i], test_components_tensor[:,i], "k--")
        axs[i].scatter(test_components_tensor[:,i],y_hat[:,i], marker="x", label = "PLS prediction", color = "orange")
        
        # axs[i].set_title(f"Estimated vs Real concentration \n for {component_names[i]}")
        # axs[i].set_xlabel("true mixture")
        # axs[i].set_ylabel("estimated mixture")
        axs[i].legend()
    plt.savefig("examples/figs/varying_temperature_Wulfert/r_predicitons.pdf")
    
# %%
with torch.no_grad():
    argmin = torch.argmin(bass.beta)
    fig, axs = plt.subplots()
    axs.set_title("temperature vs lv")
    axs.scatter(training_temp_tensor, training_data.mu_x[:, argmin], c = training_components_tensor[:,1])
    axs.scatter(test_temp_tensor, test_data.mu_x[:, argmin], c = test_components_tensor[:,1], marker= "x")


# %%
args = bass.beta.argsort()
with torch.no_grad():
    fig, ax = plt.subplots(figsize=(8,6))
    training = ax.scatter(training_data.mu_x[:, args[0]], training_data.mu_x[:, args[1]], c = training_components.components[:,1], )
    test = ax.scatter(test_data.mu_x[:, args[0]], test_data.mu_x[:, args[1]], c =  test_components.get_r()[:,1], marker="x", )
    # plt.scatter(bamm.v_x[:, args[0]], bamm.v_x[:, args[1]], marker="*")
    # plt.title("Latent Variables colored \n by Water Fraction")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    legend_elements = [
        plt.Line2D([0], [0], marker='o',  label='Training Data', markerfacecolor='k', linestyle = "None", markersize=10),
        plt.Line2D([0], [0], marker='x', color='k', linestyle = "None",  label='Test Data',  markersize=10)
    ]
    plt.legend(handles = legend_elements, loc = "upper right")
    plt.savefig("examples/figs/varying_temperature_Wulfert/latent_vars_2d_snv_low_training_data_water_fraction.pdf", bbox_inches="tight")

with torch.no_grad():
    plt.figure(figsize = (8,6))
    scatter1 = plt.scatter(training_data.mu_x[:, args[0]], training_data.mu_x[:, args[1]], c=training_temp_tensor)
    scatter2 = plt.scatter(test_data.mu_x[:, args[0]], test_data.mu_x[:, args[1]], c=test_temp_tensor, marker = "x")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o',  label='Training Data', markerfacecolor='k', linestyle = "None", markersize=10),
        plt.Line2D([0], [0], marker='x', color='k', linestyle = "None",  label='Test Data',  markersize=10)
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.savefig("examples/figs/varying_temperature_Wulfert/latent_vars_2d_snv_low_training_data_temperature.pdf", bbox_inches="tight")
    plt.show()
# %%
cmap = plt.get_cmap("plasma")

training_samples = bass.get_sample_mean(training_data, [training_data, test_data])
test_samples = bass.get_sample_mean(test_data, [training_data, test_data])
with torch.no_grad():
    fig, axs = plt.subplots(3, figsize = (8,12), sharex="all")
    for i in range(3):
        axs[i].set_title(f"{component_names[i]}")
        for j in range(training_data.num_data_points):
            axs[i].plot(training_data.observations.get_inputs(),training_samples[j,:,i], 
                    color = cmap(training_temp_tensor[j].numpy()/71) ,
                    alpha = 0.5)
        for j in range(test_data.num_data_points):
            axs[i].plot(test_data.observations.get_inputs(),test_samples[j,:,i], 
                    color = cmap(test_temp_tensor[j].numpy()/71) ,
                    alpha = 0.5)
            
        axs[i].set_ylabel(f"$s_{i}(x, \lambda)$")
    axs[i].set_xlabel("wavelengths")
    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1, 
    #                 right=0.9, 
    #                 top=0.9, 
    #                 wspace=0.4, 
    #                 hspace=0.4)
    plt.subplots_adjust(
                    wspace=0.5, 
                    hspace=0.3)
    fig.savefig("examples/figs/varying_temperature_Wulfert/reconstructed_all_spectra_low_data.pdf", bbox_inches="tight")
 
# %%

# %%

covs = dirichlet_cov(test_data.components_distribution.alpha)
means = dirichlet_mean(test_data.components_distribution.alpha).detach().numpy()

 # Define the vertices of the triangle
vertices = np.array([[np.cos(-np.pi/6), np.sin(-np.pi/6) ], [np.cos(-5*np.pi/6), np.sin(-5*np.pi/6)], [np.cos(np.pi/2), np.sin(np.pi/2)]])
xx, yy = np.mgrid[np.cos(-np.pi/6):np.cos(-5*np.pi/6):500j, np.sin(-5*np.pi/6):np.sin(np.pi/2):500j]
positions = np.vstack([xx.ravel(), yy.ravel()])
fig, ax = plt.subplots(figsize = (10,10))
triangle = plt.Polygon(vertices, edgecolor='black', facecolor="None")
triangle_1 = plt.Polygon([[np.cos(-5*np.pi/6), np.sin(np.pi/2) ], [np.cos(-5*np.pi/6), np.sin(-5*np.pi/6)], [np.cos(np.pi/2), np.sin(np.pi/2)]], edgecolor='None', facecolor="white")
triangle_2 = plt.Polygon([[np.cos(np.pi/6), np.sin(np.pi/2) ], [np.cos(-np.pi/6), np.sin(-np.pi/6)], [np.cos(np.pi/2), np.sin(np.pi/2)]], edgecolor='None', facecolor="white")

n_samples = 100

proj_samples = torch.zeros([test_data.num_data_points, n_samples, 2])

for i, alpha in enumerate(test_data.components_distribution.alpha):
    dist = torch.distributions.Dirichlet(alpha)
    s = dist.sample((n_samples,))
    proj_samples[i,:, :] = project_vect_to_dirichlet(s)
proj_samples = proj_samples.reshape(-1,2)

ax.add_patch(triangle)
ax.add_patch(triangle_1)
ax.add_patch(triangle_2)
ax.set_aspect('equal')
proj_means = project_vect_to_dirichlet(means)
proj_y_hat = project_vect_to_dirichlet(y_hat)
proj_true = project_vect_to_dirichlet(all_components.numpy())
ax.scatter(proj_true[:,0], proj_true[:,1])
ax.scatter(proj_samples[:,0], proj_samples[:,1], s = 1, color = "orange", alpha = 0.5, label = "Variational \n Posterior \n Samples")
ax.scatter(proj_means[:,0], proj_means[:,1], c = "k", marker = "x", s = 20, label = "Mean")
leg = ax.legend(loc=(-0.05,0.725))
for lh in leg.legendHandles: 
    lh.set_sizes([100])
    lh.set_alpha(1)


ax.axis("off")
point = [np.cos(-np.pi/6), np.sin(-np.pi/6)  - 0.03]
ax.text(point[0], point[1], "Ethanol", ha='center', va='top',  color='black')

point = [np.cos(-5*np.pi/6), np.sin(-5*np.pi/6) - 0.03]
ax.text(point[0], point[1], "Propanol", ha='center', va='top', color='black')


point = [np.cos(np.pi/2), np.sin(np.pi/2) +0.03]
ax.text(point[0], point[1], "Water", ha='center', va='bottom', color='black')


proj_covs = project_covar_to_dirichlet(covs.detach().numpy())

# Add a title
plt.savefig("examples/figs/varying_temperature_Wulfert/dirichlet.pdf", bbox_inches = "tight")
# Display the plot
plt.show()

# %%
prob = []
for data, alpha in zip(test_components_tensor, test_data.components_distribution.alpha):
    # dist = torch.distributions.Dirichlet(torch.ones(3))
    dist = torch.distributions.Dirichlet(alpha)
    prob.append(dist.log_prob(dist.mean).detach().numpy())
    # prob.append(dist.log_prob(data).detach().numpy())

print("log_prob", np.sum(prob))
print("msep: ", torch.mean((test_data.get_r() - test_components_tensor)**2).item())
print("msep (PLS):", np.mean((y_hat - test_components_tensor.numpy())**2))
# %%

with open("examples/msep/spectroscopy.txt", 'a') as file:
    file.write(f"{torch.mean((test_data.get_r() - test_components_tensor)**2)}\n")
with open("examples/lpd/spectroscopy.txt", 'a') as file:
    file.write(f"{np.sum(prob)}\n")

# %%
