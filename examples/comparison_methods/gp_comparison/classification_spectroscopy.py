# Code required for the classification hyperspecral example in the paper

import pandas as pd
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import gpytorch
from sklearn.decomposition import PCA

from src.data import SpectralData
import argparse
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """class for the multitask GP model"""

    def __init__(self, inducing_points, num_latents, num_tasks, num_pca_dims):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_points

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_pca_dims, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main(args):

    seed = args.random_seed
    num_pca_dims = args.num_pca_dims

    fold_number = args.fold_number

    print(f"random restart {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    work_dir = Path(args.work_dir)

    # load the data
    training_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TRAIN_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)
    test_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TEST_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(training_df[0].to_numpy().reshape(-1, 1))

    training_components_one_hot = torch.Tensor(enc.transform(training_df[0].to_numpy().reshape(-1, 1)).toarray())
    training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
    test_components_one_hot = torch.Tensor(enc.transform(test_df[0].to_numpy().reshape(-1, 1)).toarray())
    test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())

    # Preprocess the data
    training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
    training_spectral_data.trim_wavelengths(0,2250)
    training_spectral_data.snv()
    test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
    test_spectral_data.trim_wavelengths(0,2250)
    test_spectral_data.snv()

    num_latents = 3 # number of latent functions to use
    num_tasks = 4 # number of tasks

    print("num_components ", num_pca_dims)

    # PCA reduce the data
    pca = PCA(n_components=num_pca_dims)
    pca.fit(training_spectral_data.spectra)
    training_spectra =torch.tensor(pca.transform(training_spectral_data.spectra))
    test_spectra = torch.tensor(pca.transform(test_spectral_data.spectra))

    test_labels = test_components_one_hot

    test_spectra = torch.tensor(pca.transform(test_spectral_data.spectra))

    X_train = training_spectra
    y_train = training_components_one_hot.argmax(dim=-1)
    X_test = test_spectra

    # Instantiate model and likelihood
    model = MultitaskGPModel(torch.rand(num_latents, 50, num_pca_dims), num_latents, num_tasks, num_pca_dims)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=num_tasks, num_features=num_tasks)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=1000)

    # Training routine
    model.train()
    likelihood.train()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    training_iterations = 2000
    
    # Train the model
    loss_list = []
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        print(i)

    if args.plot:
        with torch.no_grad():
            plt.plot(loss_list)

    model.eval()
    likelihood.eval()

    # get the predictions
    with torch.no_grad():
        test_latent_functions = model(test_spectra)
        test_preds = likelihood(test_latent_functions)
        preds_final = test_preds.probs.mean(dim=0)
        accuracy = (preds_final.argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float64).mean()
        approx_log_prob = torch.log(torch.exp(test_preds.log_prob(test_labels.argmax(dim=-1))).mean(dim=0)).sum()
        test_preds = preds_final

    # print the results
    print("accuracy:", accuracy)
    print("approx_log_prob:", approx_log_prob)
    print("final marginal log likelihood:", - loss_list[-1])
    
    for name, param in model.named_parameters():
        print(name, param)

    # save the results
    results_dict = {'seed': seed, 
    'fold_number': fold_number,
    'mll': - loss_list[-1],
    'accuracy': accuracy.item(), 
    'num_pca_dims': num_pca_dims,
    'log_prob': approx_log_prob.item()}

    experiment_name = "spectroscopy_classification_with_saved_logprob"

    save_results_csv(file_name=f'{experiment_name}.csv', 
                    path=work_dir / "results" / "GP" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'fold_{fold_number}_seed_{seed}.pt'

    file_name = f'test_log_probs'
    os.makedirs(work_dir /"results" / "GP" / "parameters" / f"{experiment_name}" , exist_ok = True)
    torch.save(test_preds, work_dir / "results" / "GP" / "parameters" / f"{experiment_name}" / f'{file_name}_{file_appendix}')

    save_parameters(path=work_dir / "results" / "GP" / "parameters" / f"{experiment_name}", 
                    file_appendix=file_appendix, 
                    model=model, 
                    training_dataset=None, 
                    test_dataset=None)
    
    save_elbos(path=work_dir / "results" / "GP" / "full_elbos" / f"{experiment_name}", 
                file_appendix=file_appendix, 
                elbos=loss)

    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()  

    save_grads(path=work_dir / "results" / "GP" / "gradients" / f"{experiment_name}",
            file_appendix=file_appendix, 
            model_dict={name: param.grad for name, param in model.named_parameters()}, 
            training_dataset_dict=None, 
            test_dataset_dict=None)

    print("gradients")
    print({name: param.grad for name, param in model.named_parameters()})

if __name__ == "__main__":
    # Log in to your W&B account
    import os

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
        "--plot",
        "-p",
        type=bool,
        default=False,
        help="plots figures",
    )

    argparser.add_argument(
        "--num_pca_dims",
        "-npd",
        type=int,
        default=3,
        help="plots figures",
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
