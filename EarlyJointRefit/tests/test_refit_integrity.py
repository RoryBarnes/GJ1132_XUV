"""Integrity checks for the early joint hierarchical refit.

The convergence test is a hard gate: a chain that has not reached the 50-tau
criterion (Foreman-Mackey et al. 2013) fails the step, it is not a warning.
"""

import json
import os

import numpy as np
import pytest

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_TAGS = ["early_withsd", "early_nosd"]


def fdictLoadJson(sName):
    """Load a JSON product from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


@pytest.mark.parametrize("sTag", SA_TAGS)
def test_chain_converged_and_finite(sTag):
    """HARD GATE: both chains exist and are finite; every emcee run reached
    50 tau and split-R-hat < 1.01; the nested run returned a finite evidence."""
    for sStem in ("jointChain", "jointChainEmcee"):
        sChainPath = os.path.join(S_DIRECTORY, f"{sStem}_{sTag}.npy")
        assert os.path.exists(sChainPath), f"missing {sStem} for {sTag}"
        daChain = np.load(sChainPath)
        assert daChain.ndim == 2 and daChain.shape[1] == 12
        assert daChain.shape[0] > 1000
        assert np.all(np.isfinite(daChain))
    dictSummary = fdictLoadJson(f"jointFitSummary_{sTag}.json")
    assert np.isfinite(dictSummary["dictNested"]["dLogEvidence"])
    dictEmcee = dictSummary["dictEmcee"]
    assert dictEmcee["bConverged"], \
        f"{sTag}: an emcee run is short of 50 tau or split-R-hat >= 1.01"
    assert all(iSteps >= 50 * dTau for iSteps, dTau
               in zip(dictEmcee["iaSteps"], dictEmcee["daTauSteps"]))


@pytest.mark.parametrize("sTag", SA_TAGS)
def test_independent_samplers_agree(sTag):
    """HARD GATE: nested-sampling and emcee posterior means agree to within
    0.3 nested-sampling standard deviations for every hyperparameter."""
    daNested = np.load(os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy"))
    daEmcee = np.load(os.path.join(S_DIRECTORY, f"jointChainEmcee_{sTag}.npy"))
    daOffset = np.abs(np.mean(daNested, axis=0) - np.mean(daEmcee, axis=0))
    assert np.all(daOffset < 0.3 * np.std(daNested, axis=0)), daOffset


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
