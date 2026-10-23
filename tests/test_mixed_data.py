"""
Tests for data that mixes different kinds of columns:
- genuinely zero-inflated (spike and slab) columns
- columns that are always 0
- columns that are 0 or a single constant (no variance in the slab)
- constant columns without any zeros
- dense (not sparse at all) columns
"""
import warnings

import numpy as np
import pytest

from sparse_data_modeling import SparseDataModel

N_SAMPLES = 4000
N_GENERATED = 40000

# column layout of the mixed data set
ZI_A, ZI_B, ALL_ZERO, ZERO_OR_CONST, CONST, DENSE_A, DENSE_B = range(7)
N_DIMS = 7
CONST_VALUE = 3.0
ZERO_OR_CONST_VALUE = 5.0


def make_mixed_data(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    # latent values of the two zero-inflated columns and the first dense column are correlated
    latent = rng.multivariate_normal(
        np.zeros(3),
        [[1.0, 0.6, 0.5],
         [0.6, 1.0, 0.3],
         [0.5, 0.3, 1.0]],
        size=n_samples,
    )
    data = np.zeros((n_samples, N_DIMS))
    data[:, ZI_A] = np.where(rng.random(n_samples) < 0.3, 2.0 + latent[:, 0], 0.0)
    data[:, ZI_B] = np.where(rng.random(n_samples) < 0.6, -1.0 + 0.5 * latent[:, 1], 0.0)
    data[:, ZERO_OR_CONST] = np.where(rng.random(n_samples) < 0.4, ZERO_OR_CONST_VALUE, 0.0)
    data[:, CONST] = CONST_VALUE
    data[:, DENSE_A] = 4.0 + 2.0 * latent[:, 2]
    data[:, DENSE_B] = rng.normal(-1.0, 0.5, n_samples)
    return data


@pytest.fixture(scope="module")
def data():
    return make_mixed_data(N_SAMPLES)


@pytest.fixture(scope="module")
def model(data):
    np.random.seed(0)
    return SparseDataModel(data)


@pytest.fixture(scope="module")
def generated(model):
    np.random.seed(1)
    return model(N_GENERATED)


def test_fit_emits_no_warnings(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SparseDataModel(data)(10)


def test_corr_is_valid_correlation_matrix(model):
    corr = model.corr
    assert corr.shape == (2 * N_DIMS, 2 * N_DIMS)
    assert np.all(np.isfinite(corr))
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
    assert np.linalg.eigvalsh(corr).min() > -1e-10
    # a value is modelled as independent of its own mask
    for i in range(N_DIMS):
        assert abs(corr[i, i + N_DIMS]) < 1e-6


def test_degenerate_dims_detected(model):
    mask_offset = N_DIMS
    expected = {
        ALL_ZERO, ALL_ZERO + mask_offset,  # no data at all
        ZERO_OR_CONST,  # slab has no variance, mask is informative
        CONST, CONST + mask_offset,  # neither carries information
        DENSE_A + mask_offset, DENSE_B + mask_offset,  # mask is always on
    }
    assert model.degenerate_dims == expected


@pytest.mark.parametrize("prob_eps", [0.0, 0.05])
def test_degenerate_dims_are_decoupled(data, prob_eps):
    model = SparseDataModel(data, prob_eps=prob_eps)
    for i in model.degenerate_dims:
        off_diagonal = np.delete(model.corr[i], i)
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-8)


def test_degenerate_columns_do_not_change_informative_block():
    data = make_mixed_data(N_SAMPLES)[:, [ZI_A, ZI_B, DENSE_A, DENSE_B]]
    n = data.shape[1]
    padding = np.column_stack([np.zeros((N_SAMPLES, 30)), np.full((N_SAMPLES, 30), 7.0)])
    base = SparseDataModel(data).corr
    padded = SparseDataModel(np.column_stack([data, padding])).corr
    m = padding.shape[1]
    informative = list(range(n)) + list(range(n + m, 2 * n + m))
    np.testing.assert_allclose(padded[np.ix_(informative, informative)], base, atol=1e-6)


def test_nonzero_probabilities_reproduced(data, generated):
    np.testing.assert_allclose((generated != 0).mean(axis=0), (data != 0).mean(axis=0), atol=0.02)


def test_slab_statistics_reproduced(data, generated):
    for dim in [ZI_A, ZI_B, DENSE_A, DENSE_B]:
        original = data[data[:, dim] != 0, dim]
        synthetic = generated[generated[:, dim] != 0, dim]
        assert synthetic.mean() == pytest.approx(original.mean(), abs=0.1 * original.std())
        assert synthetic.std() == pytest.approx(original.std(), rel=0.1)


def test_constant_columns_reproduced(generated):
    assert np.all(generated[:, ALL_ZERO] == 0.0)
    np.testing.assert_allclose(generated[:, CONST], CONST_VALUE)
    nonzero = generated[generated[:, ZERO_OR_CONST] != 0, ZERO_OR_CONST]
    np.testing.assert_allclose(nonzero, ZERO_OR_CONST_VALUE)


def test_observed_correlations_reproduced(data, generated):
    informative = [ZI_A, ZI_B, ZERO_OR_CONST, DENSE_A, DENSE_B]
    original = np.corrcoef(data[:, informative], rowvar=False)
    synthetic = np.corrcoef(generated[:, informative], rowvar=False)
    np.testing.assert_allclose(synthetic, original, atol=0.06)


def test_latent_correlations_recovered(model):
    assert model.corr[ZI_A, ZI_B] == pytest.approx(0.6, abs=0.1)
    assert model.corr[ZI_A, DENSE_A] == pytest.approx(0.5, abs=0.1)
    assert model.corr[ZI_B, DENSE_A] == pytest.approx(0.3, abs=0.1)
    assert model.corr[DENSE_A, DENSE_B] == pytest.approx(0.0, abs=0.1)


def test_regularization_keeps_degenerate_columns_explorable(data):
    """With prob_eps and std_floor (as used in an EDA), degenerate columns still get sampled around."""
    np.random.seed(3)
    model = SparseDataModel(data, prob_eps=0.05, std_floor=0.1)
    generated = model(N_GENERATED)
    p = (generated != 0).mean(axis=0)
    assert p[ALL_ZERO] == pytest.approx(0.05, abs=0.01)
    assert p[CONST] == pytest.approx(0.95, abs=0.01)
    assert generated[generated[:, CONST] != 0, CONST].std() == pytest.approx(0.1, rel=0.1)
