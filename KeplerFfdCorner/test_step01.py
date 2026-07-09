"""
Pytest validation for KeplerFfdCorner step outputs.
Validates flare MCMC samples and Kepler FFD posterior statistics.
"""

import json
import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEP_DIR = pathlib.Path("/workspace/GJ1132_XUV/KeplerFfdCorner")
MCMC_FILE = STEP_DIR / "flare_mcmc_samples.npy"
POSTERIOR_FILE = STEP_DIR / "kepler_ffd_posterior_stats.json"


# ===================================================================
# 1. File existence and non-empty checks
# ===================================================================
class TestFileExistence:
    """All expected output files exist and are non-empty."""

    @pytest.mark.parametrize("fpath", [MCMC_FILE, POSTERIOR_FILE])
    def test_file_exists(self, fpath):
        assert fpath.exists(), f"Missing output file: {fpath.name}"

    @pytest.mark.parametrize("fpath", [MCMC_FILE, POSTERIOR_FILE])
    def test_file_non_empty(self, fpath):
        assert fpath.stat().st_size > 0, f"Output file is empty: {fpath.name}"


# ===================================================================
# 2. Data format / loadability
# ===================================================================
class TestDataFormats:
    """Output files are loadable and have expected structure."""

    def test_mcmc_samples_loadable(self):
        arr = np.load(MCMC_FILE)
        assert isinstance(arr, np.ndarray)

    def test_mcmc_samples_ndim(self):
        arr = np.load(MCMC_FILE)
        assert arr.ndim == 2, (
            f"Expected 2-D MCMC sample array (samples x params), got ndim={arr.ndim}"
        )

    def test_mcmc_samples_min_rows(self):
        arr = np.load(MCMC_FILE)
        assert arr.shape[0] >= 100, (
            f"Too few MCMC samples ({arr.shape[0]}); expected at least 100"
        )

    def test_mcmc_samples_min_cols(self):
        arr = np.load(MCMC_FILE)
        assert arr.shape[1] >= 1, "MCMC samples array has zero parameter columns"

    def test_posterior_stats_loadable(self):
        with open(POSTERIOR_FILE, "r") as fh:
            data = json.load(fh)
        assert isinstance(data, dict), "Posterior stats JSON should be a dict"

    def test_posterior_stats_non_empty(self):
        with open(POSTERIOR_FILE, "r") as fh:
            data = json.load(fh)
        assert len(data) > 0, "Posterior stats dict is empty"


# ===================================================================
# 3. Numerical validity — no NaN / Inf
# ===================================================================
class TestNumericalValidity:
    """No NaN or Inf values in numerical outputs."""

    def test_mcmc_no_nan(self):
        arr = np.load(MCMC_FILE)
        assert not np.any(np.isnan(arr)), "MCMC samples contain NaN values"

    def test_mcmc_no_inf(self):
        arr = np.load(MCMC_FILE)
        assert not np.any(np.isinf(arr)), "MCMC samples contain Inf values"

    def test_posterior_no_nan_inf(self):
        with open(POSTERIOR_FILE, "r") as fh:
            data = json.load(fh)
        _check_json_numeric(data)


def _check_json_numeric(obj, path=""):
    """Recursively assert no NaN/Inf floats in a JSON structure."""
    if isinstance(obj, float):
        assert not (np.isnan(obj) or np.isinf(obj)), (
            f"NaN or Inf at JSON path '{path}'"
        )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_json_numeric(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_json_numeric(v, path=f"{path}[{i}]")


# ===================================================================
# 4. Physically reasonable ranges
# ===================================================================
class TestPhysicalRanges:
    """MCMC parameter values and posterior stats are physically plausible."""

    def test_mcmc_samples_finite_range(self):
        """All MCMC parameter values within a broad but finite range."""
        arr = np.load(MCMC_FILE)
        assert np.all(np.abs(arr) < 1e30), (
            "MCMC samples contain values exceeding 1e30 — likely unphysical"
        )

    def test_posterior_stats_values_reasonable(self):
        """Numeric values in the posterior summary are finite and bounded."""
        with open(POSTERIOR_FILE, "r") as fh:
            data = json.load(fh)
        nums = _extract_numbers(data)
        assert len(nums) > 0, "No numeric values found in posterior stats"
        for val in nums:
            assert abs(val) < 1e30, (
                f"Posterior stat value {val} exceeds 1e30 — likely unphysical"
            )

    def test_mcmc_samples_have_variance(self):
        """Each parameter column should have non-zero variance (chain mixed)."""
        arr = np.load(MCMC_FILE)
        col_std = np.std(arr, axis=0)
        assert np.all(col_std > 0), (
            "One or more MCMC parameter columns have zero variance — chain may not have mixed"
        )


def _extract_numbers(obj):
    """Return a flat list of all numeric values in a nested JSON structure."""
    nums = []
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_extract_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            nums.extend(_extract_numbers(v))
    return nums