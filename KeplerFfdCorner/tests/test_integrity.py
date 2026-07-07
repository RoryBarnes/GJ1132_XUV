import os
import json

import numpy as np
import pytest

STEP_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = STEP_DIR


@pytest.fixture
def samples():
    return np.load(os.path.join(DATA_DIR, "flare_mcmc_samples.npy"))


@pytest.fixture
def posterior_stats():
    with open(os.path.join(DATA_DIR, "kepler_ffd_posterior_stats.json"), "r") as f:
        return json.load(f)


class TestFilesExistAndNonEmpty:
    def test_samples_file_exists(self):
        path = os.path.join(DATA_DIR, "flare_mcmc_samples.npy")
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 0, "flare_mcmc_samples.npy is empty"

    def test_posterior_stats_file_exists(self):
        path = os.path.join(DATA_DIR, "kepler_ffd_posterior_stats.json")
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 0, "kepler_ffd_posterior_stats.json is empty"


class TestSamplesLoadable:
    def test_loads_as_numpy_array(self, samples):
        assert isinstance(samples, np.ndarray)

    def test_dtype_is_float(self, samples):
        assert np.issubdtype(samples.dtype, np.floating)

    def test_shape_is_2d(self, samples):
        assert samples.ndim == 2, f"Expected 2D array, got {samples.ndim}D"

    def test_has_6_parameters(self, samples):
        assert samples.shape[1] == 6, (
            f"Expected 6 parameter columns (a1,a2,a3,b1,b2,b3), got {samples.shape[1]}"
        )

    def test_has_positive_num_samples(self, samples):
        assert samples.shape[0] > 0, "No MCMC samples found"

    def test_no_nan_values(self, samples):
        assert not np.any(np.isnan(samples)), "NaN values found in MCMC samples"

    def test_no_inf_values(self, samples):
        assert not np.any(np.isinf(samples)), "Inf values found in MCMC samples"


class TestPosteriorStatsLoadable:
    def test_loads_as_dict(self, posterior_stats):
        assert isinstance(posterior_stats, dict)

    def test_has_medians_key(self, posterior_stats):
        assert "daMedians" in posterior_stats

    def test_has_covariance_key(self, posterior_stats):
        assert "daCovarianceMatrix" in posterior_stats

    def test_has_num_samples_key(self, posterior_stats):
        assert "iNumSamples" in posterior_stats

    def test_medians_length_is_6(self, posterior_stats):
        assert len(posterior_stats["daMedians"]) == 6, (
            f"Expected 6 median values, got {len(posterior_stats['daMedians'])}"
        )

    def test_medians_are_finite(self, posterior_stats):
        for i, val in enumerate(posterior_stats["daMedians"]):
            assert np.isfinite(val), f"Median[{i}] is not finite: {val}"

    def test_covariance_matrix_shape(self, posterior_stats):
        cov = np.array(posterior_stats["daCovarianceMatrix"])
        assert cov.shape == (6, 6), f"Expected (6,6) covariance matrix, got {cov.shape}"

    def test_covariance_matrix_no_nan(self, posterior_stats):
        cov = np.array(posterior_stats["daCovarianceMatrix"])
        assert not np.any(np.isnan(cov)), "NaN values in covariance matrix"

    def test_covariance_matrix_no_inf(self, posterior_stats):
        cov = np.array(posterior_stats["daCovarianceMatrix"])
        assert not np.any(np.isinf(cov)), "Inf values in covariance matrix"

    def test_covariance_matrix_is_symmetric(self, posterior_stats):
        cov = np.array(posterior_stats["daCovarianceMatrix"])
        assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"

    def test_num_samples_is_positive_int(self, posterior_stats):
        assert isinstance(posterior_stats["iNumSamples"], int)
        assert posterior_stats["iNumSamples"] > 0

    def test_num_samples_matches_npy(self, posterior_stats, samples):
        assert posterior_stats["iNumSamples"] == samples.shape[0], (
            f"iNumSamples ({posterior_stats['iNumSamples']}) != "
            f"samples array rows ({samples.shape[0]})"
        )