# Code required to make predictions of mixture fractions in oil

import torch
from src.wsgplvm import WSGPLVM
import argparse
from pathlib import Path
from itertools import chain
from src.data import Dataset, IndependentObservations, ObservedComponents,  VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads
import time

import numpy as np
import pandas as pd

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print('device:', device)

torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed
    data_idx = args.data_idx
    num_train = args.num_train # number of training points
    num_test = args.num_test # number of test points
    num_latent_dims = 5 # number of latent dimensions
    work_dir = Path(args.work_dir)
    t1= time.time()

    num_latent_inducing = args.num_latent_inducing

    print(f"random restart {seed} data index {data_idx}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if (num_train == 1000) & (num_test == 1000):
        experiment_name = f'oil_flow_regression_{num_latent_inducing}inducing'
    else:
        experiment_name = f'oil_flow_regression_train_{num_train}_test_{num_test}_{num_latent_inducing}inducing'

    path = work_dir / "results" / "WSGPLVM" / "csvs" / f'{experiment_name}.csv'
    
    # Check if run already in results file
    df = pd.read_csv(path)
    # if len(df[(df['seed'] == seed) &  (df['data_idx'] == data_idx)]) > 0:
    #     print('run already in results file')
    #     pass 
    # else:
    if True: # Commented out the check if the run has already been done
        # Load data and move to GPU
        training_spectra = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrn_{data_idx}.txt')).to(device)
        training_labels = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrnLbls_{data_idx}.txt')).type(torch.int).to(device)
        training_components = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrnFrctns_{data_idx}.txt')).to(device)
        training_components = torch.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
        
        test_spectra = torch.Tensor(np.loadtxt(fname=work_dir / f'examples/data/3phData_splits/DataTrn_{data_idx}.txt')).to(device)
        test_labels = torch.Tensor(np.loadtxt(fname=work_dir/ f'examples/data/3phData_splits/DataTrnLbls_{data_idx}.txt')).type(torch.int).to(device)
        test_components = torch.Tensor(np.loadtxt(fname=work_dir/ f'examples/data/3phData_splits/DataTrnFrctns_{data_idx}.txt')).to(device)
        test_components = torch.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])

        # Reduce data size based on num_train and num_test
        training_spectra = training_spectra[:num_train]
        training_labels = training_labels[:num_train]
        training_components = training_components[:num_train]

        test_spectra = test_spectra[:num_test]
        test_labels = test_labels[:num_test]    
        test_components = test_components[:num_test]

        # Create datasets and move to GPU
        training_spectral_data = IndependentObservations(training_spectra, device=device)
        training_components_data = ObservedComponents(training_components, device=device)

        test_spectral_data = IndependentObservations(test_spectra, device=device)
        test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3).to(device), device=device)

        print('no. of training points:', training_spectral_data.num_data_points)
        print('no. of test points:', test_spectra.shape[0])

        x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_data, num_latent_dims).to(device)

        training_dataset = Dataset(
                        training_spectral_data, 
                        training_components_data, 
                        x_init_train,
                        Sigma_x = torch.ones(training_spectral_data.num_data_points, num_latent_dims).to(device)
                        ).to(device)

        test_dataset = Dataset(
                        test_spectral_data, 
                        test_components_data, 
                        torch.zeros(test_spectral_data.num_data_points, num_latent_dims).to(device),
                        Sigma_x = torch.ones(test_spectral_data.num_data_points, num_latent_dims).to(device)
                        ).to(device)
    
        beta = (torch.rand(x_init_train.shape[1])+0.1).to(device) # latent lengthscale
        sigma2_s = ((torch.rand(1)+1)/2).to(device) # kernel variance
        v_x = torch.randn(num_latent_inducing, num_latent_dims).to(device) # inducing points

        # Create mdoel and move to GPU. Note not input dependence so no input lengthscale or inducing points
        bass = WSGPLVM(
            beta = beta, 
            sigma2 = torch.ones(1).to(device),
            sigma2_s = sigma2_s,
            v_x = v_x,
            device=device
        ).to(device)

        loss = []

        adam_training = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters()), lr = 5e-3)
        adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr = 1e-3)

        # Training loop
        loss.extend(train_bass_on_spectral_data(bass, [training_dataset], adam_training, 4000)) #4000

        noise_schedule = log_linspace(1,bass.sigma2.item(), 20)
        for s in noise_schedule:
            print(s)
            bass.sigma2 = s
            bass.to(device)
            loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters()) , 2))

        loss.extend(train_bass_on_spectral_data(bass, [training_dataset, test_dataset], adam_all, 5000))

        loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), bass.parameters()) , 2))
        
        # Train to convergence
        while torch.abs(loss[-1] - loss[-2]) > 1:
            loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), bass.parameters()) , 1))

        prob = []
        for data, alpha in zip(test_components, test_dataset.components_distribution.alpha):
            dist = torch.distributions.Dirichlet(alpha)
            prob.append(dist.log_prob(dist.mean).detach().cpu().numpy())

        # Print results
        print("log_prob", np.sum(prob))
        print("msep: ", torch.mean((test_dataset.get_r() - test_components)**2).item())
        print("msep from mode: ", torch.mean((test_dataset.get_r_mode() - test_components)**2).item())
        print("elbo: ", loss[-1])

        # Save results

        results_dict = {'seed': seed, 
                        'data_idx': data_idx,
                        'elbo': loss[-1].item(),
                        'msep': torch.mean((test_dataset.get_r() - test_components)**2).item(), 
                        'msep from mode': torch.mean((test_dataset.get_r_mode() - test_components)**2).item(),
                        'log_prob': np.sum(prob)}

        save_results_csv(file_name= f'{experiment_name}.csv', 
                        path=work_dir / "results" / "WSGPLVM" / "csvs" / experiment_name / f'{experiment_name}.csv', 
                        results_dict=results_dict)

        file_appendix = f'data_{data_idx}_seed_{seed}.pt'

        save_parameters(path=work_dir / "results" / "WSGPLVM" / "parameters" / experiment_name, 
                        file_appendix=file_appendix, 
                        model=bass, 
                        training_dataset=training_dataset, 
                        test_dataset=test_dataset)

        save_elbos(path=work_dir / "results" / "WSGPLVM" / "full_elbos"  / experiment_name,
                    file_appendix=file_appendix, 
                    elbos=loss)

        loss = - bass.elbo([training_dataset, test_dataset])  
        loss.backward()  

        save_grads(path=work_dir / "results" / "WSGPLVM" / "gradients"  / experiment_name,
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
        "--num_latent_inducing",
        "-ni",
        type=int,
        default=50,
        help="Number latent inducing points.",
    )

    argparser.add_argument(
        "--work_dir",
        "-wd",
	type=str,
        default="",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
