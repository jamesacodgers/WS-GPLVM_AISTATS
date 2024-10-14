import matplotlib.pyplot as plt
import abc
import os, sys

import torch

from src.data import SpectralData

import numpy as np


class BasePlot(abc.ABC):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.dir = self._get_dir()
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

    def save(self, file_name):
        self.fig.savefig(f"{self.dir}/{file_name}")

    def show(self):
        self.fig.show()

    def _get_dir(self):
        file = sys.argv[0]
        file = file.split("/")[-1]
        file = file.split(".")[0]
        return f"examples/figs/{file}"

    @abc.abstractmethod
    def plot(self):
        pass

    def show(self):
        self.fig.show()

# TODO: rewrite this to take kwargs 
class ScatterPlot(BasePlot):
    def __init__(self, x, y):
        super().__init__()
        self.plot(x, y)

    def plot(self, x, y, x_label=None, y_label=None, title=None):
        self.ax.scatter(x, y)
        if x_label:
            self.ax.set_xlabel(x_label)
        if y_label:
            self.ax.set_ylabel(y_label)
        if title:
            self.ax.set_title(title)


class SpectraPlot(BasePlot):
    def __init__(self, spectral_data: SpectralData):
        super().__init__()
        self.spectra = spectral_data.spectra
        self.wavelengths = spectral_data.wavelengths.flatten()

    def plot(
        self,
        alpha=1 
    ):
        with torch.no_grad():
            self.ax.plot(
                self.wavelengths,
                self.spectra.T,
                alpha=alpha,
                color="orange",
            )
            self.ax.set_xlabel("wavelength")
            self.ax.set_ylabel("spectra")
        plt.show()    

# %%
def project_vect_to_dirichlet(inputs: np.array) -> np.array:
    map = np.array([[np.cos(-5/6*np.pi), 0, np.cos(-1/6*np.pi)],
                        [np.sin(-5/6*np.pi), 1, np.sin(-1/6*np.pi)]]
                    )
    return inputs@ map.T
def project_covar_to_dirichlet(inputs: np.array) -> np.array:
    map = np.array([[np.cos(-5/6*np.pi), 0, np.cos(-1/6*np.pi)],
                        [np.sin(-5/6*np.pi), 1, np.sin(-1/6*np.pi)]]
                    )
    return map @ inputs@ map.T 

def save_vec(path: str, vec, header) -> None:
    np.savetxt(path, vec, header=header, comments="", fmt="%.6f", delimiter=" " )
    

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2

    # Use LaTeX font in matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  # You can use 'sans-serif', 'monospace', etc.

    df_true = pd.read_csv("wulfert_true.dat", delimiter=" ")
    # df_pred = pd.read_csv("wulfert_est.dat", delimiter=" ", index_col=False)
    df_mean = pd.read_csv("wulfert_means", delimiter=" ", index_col=False)
    df_covs = pd.read_csv("wulfert_covs", delimiter=" ", index_col=False, header = None )
    # Define the vertices of the triangle
    vertices = np.array([[np.cos(-np.pi/6), np.sin(-np.pi/6) ], [np.cos(-5*np.pi/6), np.sin(-5*np.pi/6)], [np.cos(np.pi/2), np.sin(np.pi/2)]])

    # Create a plot and add the triangle
    fig, ax = plt.subplots(figsize = (8,8))
    triangle = plt.Polygon(vertices, edgecolor='black', facecolor="white")
    ax.add_patch(triangle)
    ax.set_aspect('equal')

    # Set axis limits
    # ax.set_xlim(-1.02, 2.02)
    # ax.set_ylim(-1.04 *  np.sqrt(3), np.sqrt(3)/2*1.04 )

    # Set axis labels

    ax.axis("off")
    point = [np.cos(-np.pi/6), np.sin(-np.pi/6)  - 0.03]
    ax.text(point[0], point[1], "Ethanol", ha='center', va='top', fontsize=24, color='black')

    point = [np.cos(-5*np.pi/6), np.sin(-5*np.pi/6) - 0.03]
    ax.text(point[0], point[1], "Propanol", ha='center', va='top', fontsize=24, color='black')


    point = [np.cos(np.pi/2), np.sin(np.pi/2) +0.03]
    ax.text(point[0], point[1], "Water", ha='center', va='bottom', fontsize=24, color='black')

    proj_means = project_vect_to_dirichlet(df_mean.to_numpy())
    plt.scatter(df_true["X"], df_true["Y"])
    plt.scatter(proj_means[:,0], proj_means[:,1], marker="x", color = "orange")

    covs = df_covs.to_numpy().reshape(-1,3,3)
    proj_covs = project_covar_to_dirichlet(covs)

    for cov,mean in zip(proj_covs, proj_means):
        theta = np.linspace(0, 2*np.pi, 20)
        circle_points = np.column_stack((np.cos(theta), np.sin(theta)))

        # Cholesky decomposition of the covariance matrix
        cholesky_matrix = np.linalg.cholesky(cov)

        # Scale the circle points by the Cholesky matrix to get ellipse points
        ellipse_points = mean + circle_points

        # Scale the circle points by the Cholesky matrix to get ellipse points
        ellipse_points = mean + 2 * np.dot(circle_points, cholesky_matrix.T)

        # Calculate the chi-squared critical value for the confidence level
        # chi2_critical = chi2.ppf(alpha, df=2)

        # Scale the ellipse points by the square root of the chi-squared critical value
        # ellipse_points *= np.sqrt(chi2_critical)   /
        plt.plot(ellipse_points[:,0], ellipse_points[:,1], color = "orange")
    # Add a title
    plt.savefig("examples/figs/varying_temperature_Wulfert/dirichlet.pdf", bbox_inches = "tight")
    # Display the plot
    plt.show()
    # %%
