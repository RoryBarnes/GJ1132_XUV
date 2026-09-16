import os
import json
import pytest
import numpy as np

STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def posterior_stats():
    sPath = os.path.join(STEP_DIR, "kepler_ffd_posterior_stats.json")
    with open(sPath, "r") as f:
        return json.load(f)


def test_posterior_stats_has_daMedians(posterior_stats):
    assert "daMedians" in posterior_stats


def test_posterior_stats_has_daCovarianceMatrix(posterior_stats):
    assert "daCovarianceMatrix" in posterior_stats


def test_posterior_stats_has_iNumSamples(posterior_stats):
    assert "iNumSamples" in posterior_stats