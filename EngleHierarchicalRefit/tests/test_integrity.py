"""Integrity checks for the Engle hierarchical refit outputs."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fdaLoad(sName):
    """Load a saved numpy array from the step directory."""
    return np.load(os.path.join(S_DIRECTORY, sName))


def test_coefficient_samples_shape_and_finite():
    """Both coefficient chains have six columns and are finite."""
    for sName in ["rotationCoefficientSamples.npy", "xuvCoefficientSamples.npy"]:
        daChain = fdaLoad(sName)
        assert daChain.ndim == 2 and daChain.shape[1] == 6
        assert daChain.shape[0] > 1000
        assert np.all(np.isfinite(daChain))


def test_intrinsic_scatter_positive():
    """Inferred segment intrinsic scatters are strictly positive."""
    for sName in ["rotationCoefficientSamples.npy", "xuvCoefficientSamples.npy"]:
        daChain = fdaLoad(sName)
        assert np.all(np.exp(daChain[:, 4]) > 0)
        assert np.all(np.exp(daChain[:, 5]) > 0)


def test_covariance_matrices():
    """Both coefficient covariance matrices are 4x4 and symmetric."""
    for sName in ["rotationCovariance.npy", "xuvCovariance.npy"]:
        daCovariance = fdaLoad(sName)
        assert daCovariance.shape == (4, 4)
        assert np.allclose(daCovariance, daCovariance.T)


def test_summary_reports_scatter():
    """The refit summary records a positive old-track scatter per relation."""
    with open(os.path.join(S_DIRECTORY, "refitSummary.json")) as fileHandle:
        dictSummary = json.load(fileHandle)
    for sRelation in ["rotation", "xuv"]:
        assert dictSummary[sRelation]["sigma_int_old"]["mean"] > 0
