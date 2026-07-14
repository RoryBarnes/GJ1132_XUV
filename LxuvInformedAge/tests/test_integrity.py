"""Integrity checks for the GJ 1132 activity-consistency check outputs."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fdictLoadSummary():
    """Load the consistency summary."""
    with open(os.path.join(S_DIRECTORY,
                           "activityConsistency.json")) as fileHandle:
        return json.load(fileHandle)


def test_z_offset_samples_finite_and_plausible():
    """The z posterior is finite, ample, and lives inside the prior's reach."""
    daZ = np.loadtxt(os.path.join(S_DIRECTORY, "zOffsetSamples.txt"))
    assert daZ.size > 10000
    assert np.all(np.isfinite(daZ))
    assert abs(np.mean(daZ)) < 4.0
    assert 0.0 < np.std(daZ) < 1.5


def test_summary_is_internally_consistent():
    """Summary offsets match the sample file and report a real shrinkage."""
    dictSummary = fdictLoadSummary()
    daZ = np.loadtxt(os.path.join(S_DIRECTORY, "zOffsetSamples.txt"))
    assert abs(dictSummary["z_offset"]["mean"] - np.mean(daZ)) < 1e-6
    assert dictSummary["z_offset"]["std"] < 1.0
    assert dictSummary["consistency_sigma"] >= 0.0
    assert dictSummary["sigma_int_at_rotation_age"] > 0.0
