# Code to run the near infra-red spectroscopy regression example

import torch
from src.mogplvm import MOGPLVM
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
import copy
import numpy as np
from scipy.io import loadmat
import pandas as pd
import time

from src.data import Dataset, IndependentObservations,  SpectralData, ObservedComponents,  VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data
from sklearn.cross_decomposition import PLSRegression
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)

def main(args):

    t1 = time.time()

    seed = args.random_seed
    data_idx = args.data_idx
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_latent_inducing = 6 # number of latent inducing points
    num_wavelength_inducing = 28 # number of inducing points for the wavelengths
    num_latent_dims = 2 # number of latent dimensions
  
    path = work_dir / "examples/data/flodat_splits"

    training_spectra = torch.Tensor(np.loadtxt(fname=path / f'training_spectra_{data_idx}.txt'))
    training_temp_tensor = torch.Tensor(np.loadtxt(fname=path / f'training_temp_{data_idx}.txt')).type(torch.int)
    training_components = torch.Tensor(np.loadtxt(fname=path / f'training_components_{data_idx}.txt'))
    test_spectra = torch.Tensor(np.loadtxt(fname=path / f'test_spectra_{data_idx}.txt'))
    test_temp_tensor = torch.Tensor(np.loadtxt(fname=path / f'test_temp_{data_idx}.txt')).type(torch.int)
    test_components = torch.Tensor(np.loadtxt(fname=path / f'test_components_{data_idx}.txt'))

    wavelengths = torch.Tensor(np.loadtxt(fname=path / f'wl_{seed}.txt')).reshape(-1,1)

    training_spectral_data = SpectralData(wavelengths, training_spectra)
    training_components_data = ObservedComponents(training_components)

    test_spectral_data = SpectralData(wavelengths, test_spectra)
    test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3))

    x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_data, num_latent_dims)
    
    training_dataset = Dataset(
                    training_spectral_data, 
                    training_components_data, 
                    x_init_train,
                    Sigma_x = 1*torch.ones(training_spectral_data.num_data_points, num_latent_dims
                    ))
            
    test_dataset = Dataset(
                    test_spectral_data, 
                    test_components_data, 
                    # x_test_init,
                    torch.zeros(test_spectral_data.num_data_points, num_latent_dims),
                    Sigma_x = 1*torch.ones(test_spectral_data.num_data_points, num_latent_dims
                    ))

    beta = (torch.rand(x_init_train.shape[1])+0.1) *5 # latent lengthscale

    sigma2_s = (torch.rand(1)+1)/2 # kernel variance
    v_x = torch.randn(num_latent_inducing, num_latent_dims) # inducing points for the latent space

    # if including input dependence, we include the wavelengths as an input to the model
    if args.include_wl:
        gamma = (torch.rand(1)+0.01) *1000 # input lengthscale
        v_l = torch.linspace(wavelengths.min(), wavelengths.max(), num_wavelength_inducing).reshape(-1,1) # inducing points for the wavelengths
    else:
        gamma=None
        v_l=None
        

    bass = MOGPLVM(
        beta = beta, 
        gamma=gamma, 
        sigma2 = torch.ones(1)*1,
        sigma2_s = sigma2_s,
        v_x = v_x,
        v_l=v_l
    )

    loss = []
    # Optimise the model

    adam_bass = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters()), 5e-2)
   
    loss.extend(train_bass_on_spectral_data(bass, [training_dataset], adam_bass, 1000))

    test_loss = []

    sigma_2 = copy.deepcopy(bass.sigma2.item())

    noise_schedule = log_linspace(1, sigma_2, 20)
    for i in range(1):
        for s in noise_schedule:
            print(s)
            bass.sigma2 = s
            test_loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset],  chain(test_dataset.parameters(), training_dataset.parameters(),), 2))
 
    test_loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), bass.parameters()), 2)) 
            
    while test_loss[-1] - test_loss[-2] > 1 :
        test_loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), bass.parameters()), 1)) 


    prob = []
    for data, alpha in zip(test_components, test_dataset.components_distribution.alpha):
        dist = torch.distributions.Dirichlet(alpha)
        prob.append(dist.log_prob(dist.mean).detach().numpy())

    # print results

    print("log_prob", np.sum(prob))
    print("msep: ", torch.mean((test_dataset.get_r() - test_components)**2).item())
    print("msep from mode: ", torch.mean((test_dataset.get_r_mode() - test_components)**2).item())
    print("elbo train: ", loss[-1])
    print("elbo test: ", test_loss[-1])

    t2 = time.time()
    print('run time', t2-t1)

    # save results
    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'elbo train': loss[-1],
    'elbo test': test_loss[-1].item(),
    'msep': torch.mean((test_dataset.get_r() - test_components)**2).item(), 
    'msep from mode': torch.mean((test_dataset.get_r_mode() - test_components)**2).item(),
    'log_prob': np.sum(prob)}

    if args.include_wl:
        experiment_name = f'spectroscopy_regression'

    else:
        experiment_name = f'spectroscopy_regression_no_wl'

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "MOGPLVM" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'data_{data_idx}_seed_{seed}.pt'

    save_parameters(path=work_dir / "results" / "MOGPLVM" / "parameters" / experiment_name, 
                    file_appendix=file_appendix, 
                    model=bass, 
                    training_dataset=training_dataset, 
                    test_dataset=test_dataset)

    save_elbos(path=work_dir / "results" / "MOGPLVM" / "full_elbos"  / experiment_name,
                file_appendix=file_appendix, 
                elbos=loss)

    loss = - bass.elbo([training_dataset, test_dataset])  
    loss.backward()  

    save_grads(path=work_dir / "results" / "MOGPLVM" / "gradients"  / experiment_name,
            file_appendix=file_appendix, 
            model_dict={name: param.grad for name, param in bass.named_parameters()}, 
            training_dataset_dict={name: param.grad for name, param in training_dataset.named_parameters()}, 
            test_dataset_dict={name: param.grad for name, param in test_dataset.named_parameters()})

    print("gradients")
    print({name: param.grad for name, param in bass.named_parameters()})
    print({name: param.grad for name, param in training_dataset.named_parameters()})
    print({name: param.grad for name, param in test_dataset.named_parameters()})

if __name__ == "__main__":


    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--random_seed",
        "-rs",
        type=int,
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--data_idx",
        "-d",
        type=int,
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--work_dir",
        "-wd",
	type=str,
        default="",
        help="working directory",
    )

    argparser.add_argument(
        "--include_wl",
        "-iwl",
        action='store_true',
        help="include dependence of wavelengths",
    )

    args = argparser.parse_args()

    main(args)
