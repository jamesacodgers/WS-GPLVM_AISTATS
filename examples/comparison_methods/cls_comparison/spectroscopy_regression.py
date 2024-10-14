
import torch
from src.cls import CLS
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
import numpy as np

from src.data import Dataset, SpectralData, ObservedComponents,  VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.train_utils import lbfgs_training_loop
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed
    data_idx = args.data_idx
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    path = work_dir / "examples/data/flodat_splits"

    training_spectra = torch.Tensor(np.loadtxt(fname=path / f'training_spectra_{data_idx}.txt'))
    training_components = torch.Tensor(np.loadtxt(fname=path / f'training_components_{data_idx}.txt'))
    test_spectra = torch.Tensor(np.loadtxt(fname=path / f'test_spectra_{data_idx}.txt'))
    test_components = torch.Tensor(np.loadtxt(fname=path / f'test_components_{data_idx}.txt'))

    # Load the wavelengths
    wavelengths = torch.Tensor(np.loadtxt(fname=path / f'wl_{seed}.txt')).reshape(-1,1)

    training_spectral_data = SpectralData(wavelengths, training_spectra)
    training_components_data = ObservedComponents(training_components)

    test_spectral_data = SpectralData(wavelengths, test_spectra)
    test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3))
    
    num_latent_dims = 2 # number of latent dimensions

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

    # import pdb; pdb.set_trace()
    gamma = (torch.rand(1)+0.01) *1000 # input lengthscale
    sigma2_s = (torch.rand(1)+1)/2 # kernel variance

    # Create the model
    cls = CLS(M = wavelengths.shape[0], C=3, sigma2=1., gamma=gamma, sigma2_s=sigma2_s, wavelengths=wavelengths.flatten())


    # Train the model
    test_loss = []

    test_loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), cls.parameters()), 2)) 
            
    while test_loss[-1] - test_loss[-2] > 1 :
        test_loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), cls.parameters()), 1)) 

    if args.plot:
        with torch.no_grad():
            plt.plot(test_loss)
        # %%
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",  # You can change this to "sans-serif" if needed
            "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
            "font.size": 28  # Adjust the font size as needed
        })

    # Get predictions
    prob = []
    for data, alpha in zip(test_components, test_dataset.components_distribution.alpha):
        dist = torch.distributions.Dirichlet(alpha)
        prob.append(dist.log_prob(dist.mean).detach().numpy())

    print("log_prob", np.sum(prob))
    print("msep: ", torch.mean((test_dataset.get_r() - test_components)**2).item())
    print("msep from mode: ", torch.mean((test_dataset.get_r_mode() - test_components)**2).item())
    print("elbo test: ", test_loss[-1])

    # Save results
    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'elbo test': test_loss[-1].item(),
    'msep': torch.mean((test_dataset.get_r() - test_components)**2).item(), 
    'msep from mode': torch.mean((test_dataset.get_r_mode() - test_components)**2).item(),
    'log_prob': np.sum(prob)}

    experiment_name = f'spectroscopy_regression'

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "CLS" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'data_{data_idx}_seed_{seed}.pt'

    save_parameters(path=work_dir / "results" / "CLS" / "parameters" / experiment_name, 
                    file_appendix=file_appendix, 
                    model=cls, 
                    training_dataset=training_dataset, 
                    test_dataset=test_dataset)

    save_elbos(path=work_dir / "results" / "CLS" / "full_elbos"  / experiment_name,
                file_appendix=file_appendix, 
                elbos=test_loss)

    loss = - cls.elbo([training_dataset, test_dataset])  
    loss.backward()  

    save_grads(path=work_dir / "results" / "CLS" / "gradients"  / experiment_name,
            file_appendix=file_appendix, 
            model_dict={name: param.grad for name, param in cls.named_parameters()}, 
            training_dataset_dict={name: param.grad for name, param in training_dataset.named_parameters()}, 
            test_dataset_dict={name: param.grad for name, param in test_dataset.named_parameters()})

    print("gradients")
    print({name: param.grad for name, param in cls.named_parameters()})
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
        "--plot",
        "-p",
        type=bool,
        default=False,
        help="plots figures",
    )

    argparser.add_argument(
        "--work_dir",
        "-wd",
	type=str,
        default="./",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
