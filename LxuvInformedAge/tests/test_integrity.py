"""Integrity checks for the L_XUV-informed age inference outputs."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_MAX_AGE_GYR = 13.0


def fdaLoadAgesGyr(sName):
    """Load an age-sample file and return the ages in Gyr."""
    return np.loadtxt(os.path.join(S_DIRECTORY, sName)) / 1e9


def test_age_samples_within_physical_bounds():
    """Both age posteriors are positive and truncated at 13 Gyr."""
    for sName in ["lxuvInformedAgeSamples.txt", "rotationOnlyAgeSamples.txt"]:
        daAge = fdaLoadAgesGyr(sName)
        assert daAge.size > 1000
        assert np.all(np.isfinite(daAge))
        assert np.all(daAge > 0)
        assert np.all(daAge <= D_MAX_AGE_GYR + 1e-3)


def test_stats_present_and_consistent():
    """The stats file reports finite ages and a non-negative tension."""
    with open(os.path.join(S_DIRECTORY, "ageInferenceStats.json")) as fileHandle:
        dictStats = json.load(fileHandle)
    assert dictStats["tension_sigma"] >= 0
    for sKey in ["rotation_only", "lxuv_informed", "xuv_only"]:
        dLower, dUpper = dictStats[sKey]["ci95_age_gyr"]
        assert 0 < dLower < dUpper <= D_MAX_AGE_GYR + 1e-3
