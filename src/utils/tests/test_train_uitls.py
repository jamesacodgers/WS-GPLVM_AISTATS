# import torch
# import pytest
# from src.data import SpectralData
# from src.BASS import BASS
# from src.utils.train_utils import train_bass_on_spectral_data


# @pytest.fixture(name="data")
# def data_fixture():
#     D = torch.randn(100,100)
#     R = torch.randn(100,2)
#     wavelengths = torch.arange(100).reshape(-1,1)
#     return SpectralData(wavelengths, D , R)

# @pytest.fixture(name = "bamm")
# def bamm_fixture(data):
#     torch.manual_seed(1234)
#     bamm = BASS(
#         data,
#         mu_x_init=torch.randn(100, 10),
#         Sigma_x_init=torch.ones(100,10),
#         beta=torch.ones(10)*10,
#         gamma=torch.ones(1)/10,
#         sigma2=torch.Tensor([0.01]),
#         sigma2_s=torch.ones(1)*10,
#         v = torch.randn(20, 11)
#     )
#     return bamm

# class TestBammOnSpectralData:
#     def test_simple(self, data, bamm: BASS):
#         optimiser = torch.optim.Adam(
#             params=bamm.parameters(),
#             lr=0.01, 
#         )
#         train_bass_on_spectral_data(bamm, data, optimiser, 10)


