# Code required for the classification hyperspecral example in the paper

import copy 
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import torch
import numpy as np
from src.data import SpectralData

import argparse
from src.utils.save_utils import save_results_csv


torch.set_default_dtype(torch.float64)

def main(args):

    seed = args.random_seed

    fold_number = args.fold_number

    print(f"random restart {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    work_dir = Path(args.work_dir)

    # Load data
    training_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TRAIN_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)
    test_df = pd.read_csv(work_dir / f"examples/data/UCRArchive_2018/Rock/cross_val/Rock_TEST_diff_fold_{fold_number}.tsv", delimiter="\t", header  = None)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(training_df[0].to_numpy().reshape(-1, 1))

    training_components_one_hot = enc.transform(training_df[0].to_numpy().reshape(-1, 1)).toarray()
    training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
    test_components_one_hot = enc.transform(test_df[0].to_numpy().reshape(-1, 1)).toarray()
    test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())

    training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
    training_spectral_data.trim_wavelengths(0,2250)
    training_spectral_data.snv()
    test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
    test_spectral_data.trim_wavelengths(0,2250)
    test_spectral_data.snv()


    train_x = training_spectral_data.spectra
    train_y = training_components_one_hot

    test_x = test_spectral_data.spectra
    test_y = test_components_one_hot


    # Cross validation to determine dimensions
    nA = 12
    msep = np.zeros([nA])

    kf = KFold(n_splits=10, shuffle=True)

    for i in range(nA):
        for train_idx, test_idx in kf.split(train_x):
            pls_reg = PLSRegression(i+1)
            pls_reg.fit(train_x[train_idx], train_y[train_idx])
            # y_hat = pls_reg.predict(test_spectra.spectra)
            y_hat = pls_reg.predict(train_x[test_idx])
            msep[i] += ((y_hat - train_y[test_idx])**2).sum()

    min_err_components = np.argmin(msep)

    # Train model with optimal number of components
    pls_reg = PLSRegression(min_err_components.item()+1)
    pls_reg.fit(train_x, train_y)
    y_hat = pls_reg.predict(test_x)
    fig, ax = plt.subplots(3)

    if args.plot:
        for i in range(3):
            ax[i].scatter(test_y, y_hat)
        plt.show()

    accuracy = np.sum(np.argmax(y_hat,axis=1) == np.argmax(test_y, axis = 1))/test_y.shape[0]

    print("accuracy:", accuracy)
    print("cross val error:", min_err_components)

    # Save Results

    results_dict = {'seed': seed, 
    'fold_number': fold_number,
    'accuracy': accuracy, 
    'cross_val_error': min_err_components}

    save_results_csv(file_name='spectroscopy_classification.csv', 
                    path=work_dir / "results" / "PLS" / "csvs" / 'spectroscopy_classification.csv', 
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
        "--work_dir",
        "-wd",
	type=str,
        default="",
        help="working directory",
    )

    args = argparser.parse_args()

    main(args)


# %%
