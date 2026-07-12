"""Integrity checks for the L_XUV-informed cumulative XUV flux outputs."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fdaLoadFlux(sName):
    """Load a cumulative-XUV-flux sample file from the step directory."""
    return np.loadtxt(os.path.join(S_DIRECTORY, sName))


def test_flux_samples_positive_and_finite():
    """Both propagated flux distributions are positive and finite."""
    for sName in ["cumulativeXuvFluxSamplesInformed.txt",
                  "cumulativeXuvFluxSamplesRotationOnly.txt",
                  "cumulativeXuvFluxSamplesFlaresInformed.txt",
                  "cumulativeXuvFluxSamplesFlaresRotationOnly.txt"]:
        daFlux = fdaLoadFlux(sName)
        assert daFlux.size > 100
        assert np.all(np.isfinite(daFlux))
        assert np.all(daFlux > 0)


def test_stats_report_bounded_means():
    """The stats file reports ordered CIs bracketing the mean for each model."""
    with open(os.path.join(S_DIRECTORY, "cumulativeXuvStats.json")) as fileHandle:
        dictStats = json.load(fileHandle)
    for sKey in ["EngleLxuvInformed", "EngleRotationOnly",
                 "EngleBarnesLxuvInformed", "EngleBarnesRotationOnly"]:
        dLower, dUpper = dictStats[sKey]["ci95"]
        assert dLower < dictStats[sKey]["mean"] < dUpper
