
# %%

import torch
import argparse
from pathlib import Path
import numpy as np
import gpytorch
from sklearn.decomposition import PCA

from src.data import SpectralData, ObservedComponents,  VariationalDirichletDistribution
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_latents, num_tasks):
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
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
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
    data_idx = args.data_idx
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    path = work_dir / "examples/data/flodat_splits"

    test_type = "_75percenttest"

    training_spectra = torch.Tensor(np.loadtxt(fname=path / f'training_spectra{test_type}_{data_idx}.txt'))
    training_temp_tensor = torch.Tensor(np.loadtxt(fname=path / f'training_temp{test_type}_{data_idx}.txt')).type(torch.int)
    training_components = torch.Tensor(np.loadtxt(fname=path / f'training_components{test_type}_{data_idx}.txt'))
    test_spectra = torch.Tensor(np.loadtxt(fname=path / f'test_spectra{test_type}_{data_idx}.txt'))
    test_temp_tensor = torch.Tensor(np.loadtxt(fname=path / f'test_temp{test_type}_{data_idx}.txt')).type(torch.int)
    test_components = torch.Tensor(np.loadtxt(fname=path / f'test_components{test_type}_{data_idx}.txt'))

    wavelengths = torch.Tensor(np.loadtxt(fname=path / f'wl_{seed}.txt')).reshape(-1,1)

    training_spectral_data = SpectralData(wavelengths, training_spectra)
    training_components_data = ObservedComponents(training_components)

    test_spectral_data = SpectralData(wavelengths, test_spectra)
    test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3))
    
    cov_matrix = (torch.matmul(training_spectral_data.spectra.T, training_spectral_data.spectra)  / (training_spectral_data.num_data_points - 1))

    # Perform Singular Value Decomposition
    U, S, V = torch.svd(cov_matrix)

    # Calculate cumulative variance
    total_variance = torch.sum(S)
    cumulative_variance = torch.cumsum(S, 0)/total_variance

    # Find the number of components to reach 99% of variance
    num_pca_dims = 1 
    while (cumulative_variance[num_pca_dims-1] < 1 - 1e-4): 
        num_pca_dims += 1 

    print("num_components ", num_pca_dims)

    # training_spectra, S, V = torch.pca_lowrank(training_spectral_data.spectra, num_pca_dims)

    # test_spectra = torch.matmul(test_spectral_data.spectra, V)

    pca = PCA(n_components=num_pca_dims)
    pca.fit(training_spectral_data.spectra)
    training_spectra = pca.transform(training_spectral_data.spectra)

    train_x = torch.tensor(training_spectra)
    train_y = training_components
    test_x = torch.tensor(pca.transform(test_spectra))
    test_y = test_components

    num_latents = 2
    num_tasks = 3

    model = MultitaskGPModel(torch.rand(num_latents, 50, num_pca_dims), num_latents, num_tasks)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    output = model(train_x)
    print(output.__class__.__name__, output.event_shape)

    num_epochs = 2000

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=1e-2)

    # # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
    # effective for VI.
    epochs_iter = range(num_epochs)
    loss_list = []
    for i in epochs_iter:
        print(i)
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    print("log_prob:", predictions.log_prob(test_y))
    print("msep:", torch.mean((mean - test_y)**2).item())
    print("loss: ", loss_list[-1])
    print("num_components ", num_pca_dims)

    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'mll': loss_list[-1],
    'msep': torch.mean((mean - test_y)**2).item(),
    'log_prob': predictions.log_prob(test_y).item()}

    experiment_name = f'spectroscopy_regression{test_type}'

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "GP" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'data_{data_idx}_seed_{seed}.pt'

    save_parameters(path=work_dir / "results" / "GP" / "parameters" / experiment_name, 
                    file_appendix=file_appendix, 
                    model=model, 
                    training_dataset=None, 
                    test_dataset=None)

    save_elbos(path=work_dir / "results" / "GP" / "full_elbos"  / experiment_name,
                file_appendix=file_appendix, 
                elbos=loss_list)

    output = model(train_x)
    ml = -mll(output, train_y)
    ml.backward()  

    save_grads(path=work_dir / "results" / "GP" / "gradients"  / experiment_name,
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
        default="",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
