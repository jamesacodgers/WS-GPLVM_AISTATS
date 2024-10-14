# Code required for the classification hyperspecral example in the paper


from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import torch
import numpy as np


import argparse
from src.utils.save_utils import save_results_csv


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

    test_x = test_spectra
    test_y = test_components

    print('no. of training points:', train_x.shape[0])
    print('no. of test points:', test_x.shape[0])

    # Cross validation to determine number of components

    nA = 12
    msep = np.zeros([nA,3])

    d1,d2, r1, r2 =  train_test_split(train_x, train_y)

    for i in range(nA):
        pls_reg = PLSRegression(i+1)
        pls_reg.fit(d1, r1)
        # y_hat = pls_reg.predict(test_spectra.spectra)
        y_hat = pls_reg.predict(d2)
        # plt.figure()
        for j in range(3):
            # msep[i,j] = ((test_components_tensor[:,j] - y_hat[:,j])**2).sum()
            msep[i,j] = ((r2[:,j] - y_hat[:,j])**2).sum()
        print(msep)

    min_err_components = np.argmin(msep.sum(axis=1))
    print(min_err_components)

    # Train model with optimal number of components
    pls_reg = PLSRegression(min_err_components.item()+1)
    pls_reg.fit(training_spectra, training_components)
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

    if (num_train == 1000) & (num_test == 1000):
        experiment_name = 'oil_flow_regression'
    else:
        experiment_name = f'oil_flow_regression_train_{num_train}_test_{num_test}'

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


# %%
