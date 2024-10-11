# Code required for the classification hyperspecral example in the paper

# %%
import copy 
from itertools import chain
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from src.utils.save_utils import save_elbos, save_grads, save_parameters, save_results_csv
import torch
from src.cls import CLS
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import Dataset, IndependentObservations, ObservedComponents, SpectralData, VariationalClassifier, VariationalDirichletDistribution, get_init_values_for_latent_variables, predict_components_from_static_spectra
from src.utils.plot_utils import SpectraPlot
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data
import argparse

torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed
    work_dir = Path(args.work_dir)


    data_idx = args.data_seed

    print(f"random restart {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    rr = args.random_seed

    work_dir = Path.cwd()

    # Load data
    training_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TRAIN_diff_fold_{data_idx}.tsv", delimiter="\t", header  = None)
    test_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TEST_diff_fold_{data_idx}.tsv", delimiter="\t", header  = None)

    training_components_one_hot = torch.Tensor(pd.get_dummies(training_df[0]).to_numpy())
    training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
    test_components_one_hot = torch.zeros(7,4)
    for i in range(7):
        test_components_one_hot[i,test_df.iloc[i,0]-1] = 1
    test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())

    num_latent_dims = 2 

    # Preprocess data
    training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
    training_spectral_data.trim_wavelengths(0,2250)
    training_spectral_data.snv()
    test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
    test_spectral_data.trim_wavelengths(0,2250)
    test_spectral_data.snv()
    test_spectral_data = IndependentObservations( test_spectral_data.spectra)
    training_spectral_data = IndependentObservations(training_spectral_data.spectra)


    training_components_object = ObservedComponents(training_components_one_hot)
    test_components = VariationalClassifier(torch.ones_like(test_components_one_hot)/test_components_one_hot.shape[1])

    x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_object, num_latent_dims)
    x_init_test = get_init_values_for_latent_variables(test_spectral_data, training_spectral_data, training_components_object, num_latent_dims)


    training_dataset = Dataset(training_spectral_data, training_components_object, x_init_train, torch.ones_like(x_init_train))
    test_dataset = Dataset(test_spectral_data, test_components, torch.zeros_like(x_init_test), torch.ones(test_spectral_data.num_data_points, num_latent_dims))


    sigma2_s = (torch.rand(1)+1)/2 # kernel variance

    print(f"inits: sigma2_s {sigma2_s}")

    # model
    cls = CLS(
        M = training_spectral_data.num_measurements_per_datapoint,
        C=4,
        sigma2 = torch.ones(1)*1, 
        sigma2_s = sigma2_s, 
    )

    loss = []
    print('training')


    cls.elbo([training_dataset, test_dataset])
    # loss.extend(train_bass_on_spectral_data(cls, [training_dataset, test_dataset], adam_all, 1000))
    loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), cls.parameters()), 2)) 
    while loss[-1] - loss[-2] >1:
        loss.extend(lbfgs_training_loop(cls, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), cls.parameters()), 1)) 

    # Print the final parameters and results
    for parameter_item in [cls, training_dataset, test_dataset]:
        for name, param in parameter_item.named_parameters():
            print(name, param)

    print(test_dataset.get_r())
    accuracy = (test_dataset.get_r().argmax(axis=1) == test_components_one_hot.argmax(axis=1)).sum()
    print(f"random restart {rr}")
    print("accuracy is ", (accuracy/ test_dataset.num_data_points).item())

    print("log_prob:", torch.log(test_dataset.get_r()[test_components_one_hot.type(torch.bool)]).sum())
    print("final loss:", loss[-1])


    prob = []
    for data, alpha in zip(test_components_one_hot, test_dataset.components_distribution.get_r()):
        dist = torch.distributions.Categorical(alpha)
        prob.append(dist.log_prob(data.argmax()).detach().numpy())
    accuracy = (test_dataset.get_r().argmax(axis = 1) == test_components_one_hot.argmax(axis = 1 )).type(torch.float).mean()
    print("log_prob", np.sum(prob))
    print("accuracy: ", accuracy)
    print("elbo test: ", loss[-1])

    # Save results
    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'elbo test': loss[-1].item(),
    "accuracy": accuracy.item(),
    'log_prob': np.sum(prob)}

    experiment_name = f'spectroscopy_classification'

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
        "--data_seed",
        "-ds",
        type=int,
        default=0,
        help="Random seed.",
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

