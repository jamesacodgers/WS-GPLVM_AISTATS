import torch
from src.data import SpectralData
from src.utils.plot_utils import ScatterPlot, SpectraPlot
import os, sys
import shutil

x_test = [0, 1]
y_test = [0, 1]


class UnitOfWork:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        dir_ = sys.argv[0]
        dir_ = dir_.split("/")[-1]
        dir_ = dir_.split(".")[0]
        shutil.rmtree(f"./examples/figs/{dir_}")


class TestPlot:
    def test_save_fig(self):
        test_plot = ScatterPlot(x_test, y_test)
        uow = UnitOfWork()
        file_name = "test.png"
        dir_ = sys.argv[0]
        dir_ = dir_.split("/")[-1]
        dir_ = dir_.split(".")[0]
        with uow:
            test_plot.save(file_name)
            if not os.path.isfile(f"./examples/figs/{dir_}/test.png"):
                raise Exception("File not saved")

    def test_get_dir(self):
        dir_ = sys.argv[0]
        dir_ = dir_.split("/")[-1]
        dir_ = dir_.split(".")[0]
        test_plot = ScatterPlot(x_test, y_test)
        assert test_plot._get_dir() == f"examples/figs/{dir_}"


class TestScatterPlot:
    def test_plot(self):
        test_plot = ScatterPlot(x_test, y_test)
        test_plot.plot(x_test, y_test, "x_label", "y_label", "title")
        # test_plot.save("test.png")


class TestSpectraPlot:
    def test_plot(self):
        uow = UnitOfWork()
        with uow:
            spectral_data = SpectralData(
                torch.linspace(0, 10, 100).reshape(-1,1), torch.randn(200, 100)
            )
            test_plot = SpectraPlot(spectral_data)
            test_plot.save("test.png")
