"""Integrity checks for the joint hierarchical refit and its inputs.

The convergence test is a hard gate: a chain that has not reached the 50-tau
criterion (Foreman-Mackey et al. 2013) fails the step, it is not a warning.
"""

import json
import os
import sys

import numpy as np
import pytest

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S_DIRECTORY)

SA_TAGS = ["midlate_withsd", "midlate_nosd", "early_withsd", "early_nosd"]


def fdictLoadJson(sName):
    """Load a JSON product from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


def test_joint_samples_composition():
    """The merged samples carry the documented counts and provenance flags."""
    listMidLate = fdictLoadJson("jointSampleMidLate.json")
    assert len(listMidLate) == 119
    assert sum(d["bSubdwarf"] for d in listMidLate) == 5
    assert sum(d["bExcludedFromPaperFits"] for d in listMidLate) == 1
    iMerged = sum(d["dictXray"] is not None and d["dictRotation"] is not None
                  and d["sAgeProvenance"] == "independent" for d in listMidLate)
    assert iMerged == 5


def test_fit_filter_reproduces_paper_composition():
    """The composition filter yields the documented benchmark counts."""
    import dataRefitJointRelations as m
    assert m.fdictBuildData("midlate", False)["iBenchmarks"] == 29
    assert m.fdictBuildData("midlate", True)["iBenchmarks"] == 34
    assert m.fdictBuildData("early", False)["iBenchmarks"] == 21
    assert m.fdictBuildData("early", True)["iBenchmarks"] == 24


def test_muscles_conversion_products():
    """The re-derived conversion has a valid covariance and positive scatter."""
    dictFit = fdictLoadJson(os.path.join("musclesConversion",
                                         "conversionFit.json"))
    dictPrimary = dictFit["primary_fit_all_targets_rosat_band"]
    daCovariance = np.array(dictPrimary["covariance_slope_intercept"])
    assert daCovariance.shape == (2, 2)
    assert np.allclose(daCovariance, daCovariance.T)
    assert np.all(np.linalg.eigvalsh(daCovariance) > 0)
    dictScatter = dictPrimary["intrinsic_scatter_dex"]
    assert 0 < dictScatter["fScatterCredible16"] \
        < dictScatter["fScatterPosteriorMedian"] \
        < dictScatter["fScatterCredible84"]


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
