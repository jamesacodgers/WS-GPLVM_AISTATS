# Code required to make predictions of mixture fractions in oil

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import gpytorch


import argparse
from src.utils.save_utils import save_results_csv, save_parameters, save_elbos, save_grads


torch.set_default_dtype(torch.float64)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """class for the multitask GP model"""
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
    num_train = args.num_train
    num_test = args.num_test
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data

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
    
    train_x = training_spectra
    train_y = training_components

    print('no. of training points:', train_x.shape[0])
    print('no. of test points:', test_spectra.shape[0])

    num_latents = 3 # number of latent functions
    num_tasks = 3 # number of tasks
 
    # initialize the model with 50 inducing points. We don't PCA reduce the input here, so the 
    # input dimensions to the GP model are 12, same as the data
    model = MultitaskGPModel(torch.rand(num_latents, 50, 12), num_latents, num_tasks)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    output = model(train_x)
    print(output.__class__.__name__, output.event_shape)

    
    # Train
    num_epochs = 5000

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)


    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    epochs_iter = range(num_epochs)

    loss = []


    for i in epochs_iter:
        print(i)
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(train_x)
        ml = -mll(output, train_y)
        ml.backward()
        loss.append(ml.item())
        optimizer.step()
    
    #%%
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        test_x = test_spectra
        test_y = test_components
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    if args.plot:
        fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
        for task, ax in enumerate(axs):
            print(task)
            # Plot training data as black stars
            axs[task].scatter(test_components[:,task], mean[:,task])
            # axs[task].scatter(train_y.detach().numpy()[:,task], train_y.detach().numpy()[:,task], color='black', marker='*', zorder=10)
            axs[task].set_title(f'Task {task + 1}')

    # Print results
    print("log_prob", predictions.log_prob(test_y))
    print("msep: ", torch.mean((mean - test_y)**2).item())
    print("elbo: ", loss[-1])

    # Save results
    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'mll': loss[-1],
    'msep': torch.mean((mean - test_y)**2).item(),
    'log_prob': predictions.log_prob(test_y).item()}

    if (num_train == 1000) & (num_test == 1000):
        experiment_name = 'oil_flow_regression'
    else:
        experiment_name = f'oil_flow_regression_train_{num_train}_test_{num_test}'


    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "ILMC" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

    file_appendix = f'data_{data_idx}_seed_{seed}.pt'

    save_parameters(path=work_dir / "results" / "ILMC" / "parameters" / experiment_name, 
                    file_appendix=file_appendix, 
                    model=model, 
                    training_dataset=None, 
                    test_dataset=None)

    save_elbos(path=work_dir / "results" / "ILMC" / "full_elbos"  / experiment_name,
                file_appendix=file_appendix, 
                elbos=loss)

    output = model(train_x)
    ml = -mll(output, train_y)
    ml.backward()  

    save_grads(path=work_dir / "results" / "ILMC" / "gradients"  / experiment_name,
            file_appendix=file_appendix, 
            model_dict={name: param.grad for name, param in model.named_parameters()}, 
            training_dataset_dict=None,
            test_dataset_dict=None)

    print("gradients")
    print({name: param.grad for name, param in model.named_parameters()})

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
        default="",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
