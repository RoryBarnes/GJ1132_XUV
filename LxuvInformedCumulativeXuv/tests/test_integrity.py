"""Integrity checks for the z-offset cumulative-flux comparison outputs."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_SAMPLE_FILES = [
    "cumulativeXuvFluxSamplesZPopulation.txt",
    "cumulativeXuvFluxSamplesZInformed.txt",
    "cumulativeXuvFluxSamplesFlaresZPopulation.txt",
    "cumulativeXuvFluxSamplesFlaresZInformed.txt",
]


def test_flux_samples_positive_and_finite():
    """Every flux distribution is non-empty, finite, and positive."""
    for sName in SA_SAMPLE_FILES:
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sName))
        assert daFlux.size > 100
        assert np.all(np.isfinite(daFlux))
        assert np.all(daFlux > 0)


def test_stats_cover_all_variants():
    """The summary reports coherent intervals for all four variants."""
    with open(os.path.join(S_DIRECTORY,
                           "cumulativeXuvStats.json")) as fileHandle:
        dictStats = json.load(fileHandle)
    for sKey in ("EngleRotationOnly", "EngleLxuvInformed",
                 "EngleBarnesRotationOnly", "EngleBarnesLxuvInformed"):
        dictEntry = dictStats[sKey]
        assert dictEntry["ci95"][0] < dictEntry["mean"] < dictEntry["ci95"][1]
