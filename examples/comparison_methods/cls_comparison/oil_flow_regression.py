# Code required to make predictions of mixture fractions in oil
# %%


import torch
from src.cls import CLS
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
from src.data import Dataset, IndependentObservations, ObservedComponents,  VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data
from sklearn.cross_decomposition import PLSRegression
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


import numpy as np
import pandas as pd

torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed
    data_idx = args.data_idx
    num_train = args.num_train
    num_test = args.num_test
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    training_spectra = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrn_{data_idx}.txt'))
    training_labels = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrnLbls_{data_idx}.txt')).type(torch.int)
    training_components = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrnFrctns_{data_idx}.txt'))
    training_components = torch.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
    test_spectra = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrn_{data_idx}.txt'))
    test_labels = torch.Tensor(np.loadtxt(fname=work_dir/ f'examples/data/3phData_splits/DataTrnLbls_{data_idx}.txt')).type(torch.int)
    test_components = torch.Tensor(np.loadtxt(fname=work_dir/ f'examples/data/3phData_splits/DataTrnFrctns_{data_idx}.txt'))
    test_components = torch.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])

    training_spectra = training_spectra[:num_train]
    training_labels = training_labels[:num_train]
    training_components = training_components[:num_train]

    test_spectra = test_spectra[:num_test]
    test_labels = test_labels[:num_test]    
    test_components = test_components[:num_test]


    training_spectral_data = IndependentObservations(training_spectra)
    training_components_data = ObservedComponents(training_components)

    test_spectral_data =IndependentObservations(test_spectra)
    test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3))

    print('no. of training points:', training_spectral_data.num_data_points)
    print('no. of test points:', test_spectra.shape[0])

    num_latent_dims = 10
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


    num_latent_inducing = 16

    cls = CLS(
        M=12,
        C=3
    )

    loss = []

    adam_training = torch.optim.Adam(chain(cls.parameters(), training_dataset.parameters()), lr = 5e-3)
    adam_all = torch.optim.Adam(chain(cls.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr = 1e-3)


    
    loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), cls.parameters()) , 2))
    while  torch.absolute(loss[-1] - loss[-2]) > 1:
        loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), cls.parameters()) , 1))


    if args.plot:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",  # You can change this to "sans-serif" if needed
        "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
        "font.size": 20  # Adjust the font size as needed
        })
        args = cls.beta.argsort()
        with torch.no_grad():
            plt.scatter(training_dataset.mu_x[:,args[0]], training_dataset.mu_x[:,args[1]], c = training_labels.argmax(dim=1))
            plt.scatter(test_dataset.mu_x[:,args[0]], test_dataset.mu_x[:,args[1]], c = training_labels.argmax(dim=1), marker="x")
            # plt.scatter(bass.v_x[:,args[0]], bass.v_x[:,args[1]], c = "black", marker="*")
            # plt.savefig(f"examples/figs/oil_flow/mixture_latent_space.pdf", bbox_inches = "tight")
            theta = torch.arange(0, 2*torch.pi, 0.01)
            d = torch.sqrt(5**2/((torch.cos(theta)**2) + (torch.sin(theta)**2)) )
            # plt.plot(d*np.cos(theta), d*np.sin(theta))

    prob = []
    for data, alpha in zip(test_components, test_dataset.components_distribution.alpha):
        # dist = torch.distributions.Dirichlet(torch.ones(3))
        dist = torch.distributions.Dirichlet(alpha)
        prob.append(dist.log_prob(dist.mean).detach().numpy())
        # prob.append(dist.log_prob(data).detach().numpy())

    print("log_prob", np.sum(prob))
    print("msep: ", torch.mean((test_dataset.get_r() - test_components)**2).item())
    print("msep from mode: ", torch.mean((test_dataset.get_r_mode() - test_components)**2).item())
    print("elbo: ", loss[-1])

    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'elbo': loss[-1].item(),
    'msep': torch.mean((test_dataset.get_r() - test_components)**2).item(), 
    'msep from mode': torch.mean((test_dataset.get_r_mode() - test_components)**2).item(),
    'log_prob': np.sum(prob)}

    if (num_train == 1000) & (num_test == 1000):
        experiment_name = 'oil_flow_regression'
    else:
        experiment_name = f'oil_flow_regression_train_{num_train}_test_{num_test}'

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "CLS" / "csvs" / experiment_name / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'data_{data_idx}_seed_{seed}.pt'

    save_parameters(path=work_dir / "results" / "CLS" / "parameters" / experiment_name, 
                    file_appendix=file_appendix, 
                    model=cls, 
                    training_dataset=training_dataset, 
                    test_dataset=test_dataset)

    save_elbos(path=work_dir / "results" / "CLS" / "full_elbos"  / experiment_name,
                file_appendix=file_appendix, 
                elbos=loss)

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
        "--num_train",
        "-ntr",
        type=int,
        default=1000,
        help="Number of training points.",
    )

    argparser.add_argument(
        "--num_test",
        "-nte",
        type=int,
        default=1000,
        help="Number of test points.",
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
        default="/Users/jo816/github/Submission",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
