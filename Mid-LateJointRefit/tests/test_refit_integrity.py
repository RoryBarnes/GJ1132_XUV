"""Integrity checks for the mid-late joint hierarchical refit.

The convergence test is a hard gate: a chain that has not reached the 50-tau
criterion (Foreman-Mackey et al. 2013) fails the step, it is not a warning.
"""

import json
import os

import numpy as np
import pytest

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_TAGS = ["midlate_withsd", "midlate_nosd"]


def fdictLoadJson(sName):
    """Load a JSON product from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


@pytest.mark.parametrize("sTag", SA_TAGS)
def test_chain_converged_and_finite(sTag):
    """HARD GATE: every fit variant converged (>= 50 tau) with finite draws."""
    sChainPath = os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy")
    assert os.path.exists(sChainPath), f"missing chain for {sTag}"
    daChain = np.load(sChainPath)
    assert daChain.ndim == 2 and daChain.shape[1] == 12
    assert daChain.shape[0] > 1000
    assert np.all(np.isfinite(daChain))
    dictSummary = fdictLoadJson(f"jointFitSummary_{sTag}.json")
    assert dictSummary["bConverged"], \
        f"{sTag}: chain shorter than 50 x autocorrelation time"
    assert dictSummary["iSteps"] > 50 * dictSummary["dTauMax"]


@pytest.mark.parametrize("sTag", SA_TAGS)
def test_scatter_law_physical(sTag):
    """Each posterior's scatter law stays positive and bounded on the data."""
    sChainPath = os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy")
    if not os.path.exists(sChainPath):
        pytest.skip(f"chain for {sTag} not yet produced")
    daChain = np.load(sChainPath)
    dictSummary = fdictLoadJson(f"jointFitSummary_{sTag}.json")
    daTauGrid = np.linspace(-1.0, 1.1, 50)
    daLogSigma = (daChain[:, 10:11] + daChain[:, 11:12]
                  * (daTauGrid - dictSummary["dPivotTau"])
                  / dictSummary["dScaleTau"])
    assert np.all(np.exp(daLogSigma) < 3.0)
    assert np.all(np.exp(np.percentile(daLogSigma, 50, axis=0)) > 1e-3)
