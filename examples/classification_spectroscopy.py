# Code required for the classification hyperspecral example in the paper

# %%
import copy 
from itertools import chain
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import torch
from src.mogplvm import MOGPLVM
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

import time
from src.data import Dataset, IndependentObservations, ObservedComponents, SpectralData, VariationalClassifier, VariationalDirichletDistribution, get_init_values_for_latent_variables, predict_components_from_static_spectra
from src.utils.plot_utils import SpectraPlot
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data
import argparse
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main(args):

    t1 = time.time()

    seed = args.random_seed

    fold_number = args.fold_number

    num_latent_inducing = 16 # number of latent inducing points
    num_wl_inducing = 20 # number of inducing points for the wavelengths
    num_latent_dims = 5 # number of latent dimensions

    print(f"random restart {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    work_dir = Path(args.work_dir)

    # Save results and parameters to csv
    if args.include_wl:
        experiment_name = f'spectroscopy_classification_incl_wavelength_latent_inducing_{num_latent_inducing}_wl_inducing_{num_wl_inducing}'
    else:
        experiment_name = f'spectroscopy_classification_inducing_{num_latent_inducing}'

    path = work_dir / "results" / "MOGPLVM" / "csvs" / f'{experiment_name}.csv'
    
    # Check if run already in results file
    df = pd.read_csv(path)
    if len(df[(df['seed'] == seed) &  (df['fold_number'] == fold_number)]) > 0:
        print('run already in results file')
        pass 

    else:

        training_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TRAIN_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)
        test_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TEST_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(training_df[0].to_numpy().reshape(-1, 1))

        # Convert data to tensors and move to device
        training_components_one_hot = torch.Tensor(enc.transform(training_df[0].to_numpy().reshape(-1, 1)).toarray()).to(device)
        training_spectra = torch.Tensor(training_df.drop(0, axis=1).to_numpy()).to(device)
        test_components_one_hot = torch.Tensor(enc.transform(test_df[0].to_numpy().reshape(-1, 1)).toarray()).to(device)
        test_spectra = torch.Tensor(test_df.drop(0, axis=1).to_numpy()).to(device)

        # Create spectral data objects
        training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1).to(device), training_spectra)
        training_spectral_data.trim_wavelengths(0, 2250)
        training_spectral_data.snv()

        test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1).to(device), test_spectra)
        test_spectral_data.trim_wavelengths(0, 2250)
        test_spectral_data.snv()

        # Create components objects and move to device
        training_components_object = ObservedComponents(training_components_one_hot, device=device)
        test_components = VariationalDirichletDistribution(torch.ones_like(test_components_one_hot).to(device) / test_components_one_hot.shape[1], device=device)


        # Initialize latent variables
        x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_object, num_latent_dims).to(device)
        x_init_test = get_init_values_for_latent_variables(test_spectral_data, training_spectral_data, training_components_object, num_latent_dims).to(device)

        # Create datasets
        training_dataset = Dataset(training_spectral_data, training_components_object, x_init_train, torch.ones_like(x_init_train).to(device)).to(device)
        test_dataset = Dataset(test_spectral_data, test_components, torch.zeros_like(x_init_test).to(device), torch.ones(test_spectral_data.num_data_points, num_latent_dims).to(device)).to(device)

        # MOGPLVM model initialization
        beta = torch.rand(x_init_train.shape[1]).to(device) + 0.1 # latent lengthscale
        sigma2_s = (torch.rand(1).to(device) + 1) / 2 # kernel variance
        v_x = torch.randn(num_latent_inducing, num_latent_dims).to(device) # inducing points for the latent variables

        if args.include_wl: 
            gamma = torch.ones(1).to(device) * 200 ** 2 # lengthscale for the wavelengths
            # inducing points for the wavelengths
            v_l = torch.linspace(training_dataset.observations.get_inputs().min(), training_dataset.observations.get_inputs().max(), num_wl_inducing).reshape(-1, 1).to(device)
        else:
            gamma = None
            v_l = None

        print(f"inits: beta {beta}, sigma2_s {sigma2_s}, v_x {v_x}")

        bass = MOGPLVM(
            beta=beta,
            gamma=gamma,
            sigma2=torch.ones(1).to(device),
            sigma2_s=sigma2_s,
            v_x=v_x,
            v_l=v_l,
            device=device
        ).to(device)

        # Optimizer
        adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr=1e-3)

        # Training loop with noise schedule
        noise_schedule = log_linspace(1, 0.2, 100)
        loss = []
        for i in range(1):
            for s in noise_schedule:
                print(s)
                bass.sigma2 = s
                loss.extend(train_bass_on_spectral_data(bass, [test_dataset, training_dataset], adam_all, 100))

        test_components_classifier = VariationalClassifier(test_dataset.get_r())
        test_dataset.components_distribution = test_components_classifier

        adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr=1e-3)
        for i in range(1):
            for s in noise_schedule:
                print(s)
                bass.sigma2 = s
                loss.extend(train_bass_on_spectral_data(bass, [test_dataset, training_dataset], adam_all, 100))

        loss.extend(train_bass_on_spectral_data(bass, [training_dataset, test_dataset], adam_all, 1000))

        while loss[-1] - loss[-2] > 1:
            loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), bass.parameters()), 1))

        # Print Parameters
        for parameter_item in [bass, training_dataset, test_dataset]:
            for name, param in parameter_item.named_parameters():
                print(name, param)

        accuracy = (test_dataset.get_r().argmax(axis=1) == test_components_one_hot.argmax(axis=1)).sum()
        print(f"random restart {seed}")
        print("accuracy is ", (accuracy / test_dataset.num_data_points).item())

        print("log_prob:", torch.log(test_dataset.get_r()[test_components_one_hot.type(torch.bool)]).sum())
        print("final loss:", loss[-1])

        t2 = time.time()
        print('run time', t2-t1)

        # Save results

        results_dict = {
            'seed': seed, 
            'fold_number': fold_number,
            'elbo': loss[-1],
            'accuracy': (accuracy / test_dataset.num_data_points).item(), 
            'log_prob': torch.log(test_dataset.get_r()[test_components_one_hot.type(torch.bool)]).sum().item()
        }

        save_results_csv(file_name=f'{experiment_name}.csv', 
                        path=work_dir / "results" / "MOGPLVM" / "csvs" / f'{experiment_name}.csv', 
                        results_dict=results_dict)

        file_appendix = f'fold_{fold_number}_seed_{seed}.pt'

        save_parameters(path=work_dir / "results" / "MOGPLVM" / "parameters" / f"{experiment_name}", 
                        file_appendix=file_appendix, 
                        model=bass, 
                        training_dataset=training_dataset, 
                        test_dataset=test_dataset)

        save_elbos(path=work_dir / "results" / "MOGPLVM" / "full_elbos" / f"{experiment_name}", 
                    file_appendix=file_appendix, 
                    elbos=loss)

        loss = - bass.elbo([training_dataset, test_dataset])  
        loss.backward()  

        save_grads(path=work_dir / "results" / "MOGPLVM" / "gradients" /  f"{experiment_name}",
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
        "--fold_number",
        "-fn",
        type=int,
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--include_wl",
        "-iwl",
        action='store_true',
        help="include dependence of wavelengths",
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


# %%
