import pytest
import torch

from src.utils.tensor_utils import (
    L2_norm_along_final_dim,
    get_pca_latent_values,
    get_static_pure_component_spectra,
    invert_covariance_matrix,
    make_grid,
    weighted_L2_diff,
    diff_between_rank_2_tensors,
    tile_final_dims_to_matrix
)


class TestDiffBetweenTensors:
    def test_shape(self):
        a = [1, 2]
        b = [1, 2, 3]
        tensor_a = torch.Tensor(a).reshape(-1, 1)
        tensor_b = torch.Tensor(b).reshape(-1, 1)
        assert diff_between_rank_2_tensors(tensor_a, tensor_b).shape == torch.Size(
            [2, 3, 1]
        )

    def test_case_1(self):
        a = [0, 0]
        b = [0, 0, 0]
        tensor_a = torch.Tensor(a).reshape(-1, 1)
        tensor_b = torch.Tensor(b).reshape(-1, 1)
        assert torch.all(
            diff_between_rank_2_tensors(tensor_a, tensor_b) == torch.zeros((2, 3, 1))
        )

    def test_case_2(self):
        torch.manual_seed(1234)
        a = torch.randn((100, 10))
        b = torch.randn((200, 10))
        diff = torch.zeros(100, 200, 10)
        for i in range(100):
            for j in range(200):
                for k in range(10):
                    diff[i, j, k] = a[i, k] - b[j, k]
        assert torch.all(diff_between_rank_2_tensors(a, b) == diff)

    def test_return_good_error_for_wrong_rank_tensor(self):
        with pytest.raises(ValueError):
            a = torch.Tensor([1, 2])
            b = torch.randn((200, 10))
            diff_between_rank_2_tensors(a, b)


class TestL2Norm:
    def test_L2_norm_case_1(self):
        a = torch.Tensor([1, 2, 3])
        assert L2_norm_along_final_dim(a) == torch.Tensor([14])

    def test_L2_norm_case_2(self):
        a = torch.Tensor([0, 0, 0])
        assert L2_norm_along_final_dim(a) == torch.Tensor([0])

    def test_L2_norm_case_3(self):
        a = torch.Tensor([-1, -2, -3])
        assert L2_norm_along_final_dim(a) == torch.Tensor([14])

    def test_weighted_L2_norm_case_1(self):
        a = torch.Tensor([1, 2, 3])
        w = torch.Tensor([1, 0, 0])
        assert torch.all(L2_norm_along_final_dim(a, w) == torch.Tensor([1]))

    def test_weighted_L2_norm_case_2(self):
        a = torch.Tensor([1, 2, 3])
        w = torch.Tensor([2, 4, 8])
        assert torch.all(L2_norm_along_final_dim(a, w) == torch.Tensor([90]))

    def test_weighted_L2_norm_error(self):
        a = torch.Tensor([1, 2, 3])
        w = torch.Tensor([-1, 0, 0])
        with pytest.raises(ValueError):
            L2_norm_along_final_dim(a, w)


class TestL2NormBetweenRank2Tensors:
    def test_shape(self):
        a = [1, 2]
        b = [1, 2, 3]
        tensor_a = torch.Tensor(a).reshape(-1, 1)
        tensor_b = torch.Tensor(b).reshape(-1, 1)
        c = weighted_L2_diff(tensor_a, tensor_b)
        assert c.shape == torch.Size([2, 3])

    def test_case_2(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1234)
        a = torch.randn((100, 10))
        b = torch.randn((200, 10))
        norm = torch.zeros(100, 200)
        for i in range(100):
            for j in range(200):
                for k in range(10):
                    norm[i, j] += (a[i, k] - b[j, k]) ** 2
        assert torch.all(torch.abs(weighted_L2_diff(a, b) - norm) < 1e-10)

    def test_case_3(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1234)
        a = torch.randn((100, 10))
        b = torch.randn((200, 10))
        weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        norm = torch.zeros(100, 200)
        for i in range(100):
            for j in range(200):
                for k in range(10):
                    norm[i, j] += (a[i, k] - b[j, k]) ** 2 * weights[k]
        assert torch.all(
            torch.abs(weighted_L2_diff(a, b, weights) - norm) < 1e-10
        )


class TestInvertCovarianceMatrix:
    def test_full_rank(self):
        torch.manual_seed(1234)
        torch.set_default_dtype(torch.float64)
        A = torch.randn(100,100)
        A = A.T @ A 
        A_inv = invert_covariance_matrix(A)
        assert torch.all(torch.isclose(A_inv@A,  torch.eye(100)))

class TestBatchKron:
    def test_case_1(self):
        a = torch.arange(6).reshape(2,3)
        b = torch.ones(4,5)
        c = a[:, :, None, None]*b[None, None, :, :]
        expected_result = torch.kron(a,b)
        result = tile_final_dims_to_matrix(c)
        assert torch.all(
            torch.isclose(expected_result, result)
        )
    def test_case_2(self):
        a = torch.randn(50,34)
        b = torch.ones(72,18)
        c = a[:, :, None, None]*b[None, None, :, :]
        expected_result = torch.kron(a,b)
        result = tile_final_dims_to_matrix(c)
        assert torch.all(
            torch.isclose(expected_result, result)
        )
    
    def test_case_3(self):
        a = torch.randn(100, 5,4)
        b = torch.ones(100, 3,2)
        c = a[:, :, :, None, None]*b[:, None, None, :, :]
        expected_results = torch.zeros(100, 15,8)
        for i in range(100):
            expected_results[i] = torch.kron(a[i],b[i])
        result = tile_final_dims_to_matrix(c)
        assert torch.all(
            torch.isclose(expected_results, result)
        )

def test_get_pca_latent_values():
    E = torch.randn(1000,100)
    mu_x = get_pca_latent_values(E,5)
    assert mu_x.shape == torch.Size([1000, 5])
    assert torch.all(torch.isclose(mu_x.var(axis = 0), torch.ones(1,1) ))

def test_make_grid():
    a_tensor = torch.randn(100,10)
    b_tensor = torch.randn(50,10)
    expected = torch.zeros(5000, 20)
    for i, a in enumerate(a_tensor):
        for j, b in enumerate(b_tensor):
            expected[i*50 + j, :10] = a
            expected[i*50 + j, 10:] = b
    val = make_grid(a_tensor, b_tensor)
    assert torch.all(val == expected)


def test_get_static_pure_component_spectra():
    D = torch.randn(100,100)
    R = torch.randn(100,3)
    S_expected = torch.inverse(R.T @ R) @ R.T @ D
    assert torch.all(S_expected == get_static_pure_component_spectra(D,R))