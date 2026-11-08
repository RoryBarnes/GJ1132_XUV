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


def test_benchmark_marginalization_matches_brute_force():
    """The importance Gauss-Hermite integral of each benchmark's latent age
    matches a 200001-point grid to within 0.01 nats, at small and large scatter."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(S_DIRECTORY), "utils"))
    import dataRefitJointRelations as refit
    dictData = refit.fdictBuildData(
        os.path.join(os.path.dirname(S_DIRECTORY), "JointSamplePreparation",
                     "jointSampleMidLate.json"), "midlate", True)
    daTau = np.linspace(-2.0, 1.3, 200001)
    for dScatter in (0.003, 0.03, 0.1):
        daTheta = np.array([0.05, -0.83, -0.046, 25.0, np.log(dScatter), 0.0,
                            -0.27, -3.21, -2.8, 0.3, -0.9, 0.4])
        tLatentPrior = refit.ftBenchmarkLatentPrior(daTheta, dictData)
        for iStar in range(dictData["iBenchmarks"]):
            daLog = refit.fdaBenchmarkLogIntegrand(daTau, iStar, daTheta, dictData,
                                                    tLatentPrior)
            dGrid = np.max(daLog) + np.log(np.trapz(np.exp(daLog - np.max(daLog)), daTau))
            assert abs(dGrid - refit.fdLogBenchmarkIntegral(
                iStar, daTheta, dictData, tLatentPrior)) < 0.01
