# Spectroscopy regression using PLS

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from src.utils.save_utils import save_results_csv

torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed
    data_idx = args.data_idx
    work_dir = Path(args.work_dir)

    print(f"random restart {seed} data index {data_idx}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    path = work_dir / "examples/data/flodat_splits"

    # Load data
    training_spectra = torch.Tensor(np.loadtxt(fname=path / f'training_spectra_{data_idx}.txt'))
    training_components = torch.Tensor(np.loadtxt(fname=path / f'training_components_{data_idx}.txt'))
    test_spectra = torch.Tensor(np.loadtxt(fname=path / f'test_spectra_{data_idx}.txt'))
    test_components = torch.Tensor(np.loadtxt(fname=path / f'test_components_{data_idx}.txt'))

    train_x = training_spectra
    train_y = training_components

    test_x = test_spectra
    test_y = test_components


    # Cross validation to determine number of components
    nA = 10
    msep = np.zeros([nA,3])

    kf = KFold(n_splits=10, shuffle=True)

    for i in range(nA):
        for train_idx, test_idx in kf.split(train_x):
            pls_reg = PLSRegression(i+1)
            pls_reg.fit(train_x[train_idx], train_y[train_idx])
            # y_hat = pls_reg.predict(test_spectra.spectra)
            y_hat = pls_reg.predict(train_x[test_idx])
            msep[i] += ((y_hat - train_y.numpy()[test_idx])**2).sum()

    min_err_components = np.argmin(msep.sum(axis=1))

    # Fit the model with the optimal number of components
    pls_reg = PLSRegression(min_err_components.item()+1)
    pls_reg.fit(train_x, train_y)
    y_hat = pls_reg.predict(test_x)

    if args.plot:
        fig, ax = plt.subplots(3)
        for i in range(3):
            ax[i].scatter(test_y, y_hat)
        plt.show()

    print("msep:", ((y_hat - test_y.numpy())**2).mean())
    print("cross val error:", min_err_components)

    # Save Results

    results_dict = {'seed': seed, 
    'data_idx': data_idx,
    'msep': ((y_hat - test_y.numpy())**2).mean(), 
    'cross_val_error': min_err_components}

    experiment_name = f'spectroscopy_regression'

    save_results_csv(file_name=f'{experiment_name}.csv', 
                    path=work_dir / "results" / "PLS" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)

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
        default="",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)
