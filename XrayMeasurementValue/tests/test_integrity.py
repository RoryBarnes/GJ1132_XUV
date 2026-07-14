"""Integrity checks for the cumulative-XUV variance budget.

Two gates are load-bearing. The analytic propagation must reproduce the
vconverge forward-model sweep in the population case, or its informed-case
prediction cannot be trusted. And the level-1 decomposition must remain
near-linear (high R^2), or the additive partition of variance across input
blocks is not valid.
"""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_SAMPLE_FILES = ["fluxSamplesPopulationZ.txt", "fluxSamplesInformedZ.txt",
                   "fluxSamplesForwardModelSweep.txt"]
S_POPULATION_SOURCE = "stellar population (intrinsic scatter)"
S_CONVERSION_SOURCE = "band conversion (intrinsic scatter)"


def fdictLoadBudget():
    """Load the variance-budget product."""
    with open(os.path.join(S_DIRECTORY,
                           "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def test_flux_samples_positive_and_finite():
    """Every flux distribution is populated, finite, and positive."""
    for sName in SA_SAMPLE_FILES:
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sName))
        assert daFlux.size > 100
        assert np.all(np.isfinite(daFlux))
        assert np.all(daFlux > 0)


def test_analytic_propagation_matches_forward_model():
    """HARD GATE: analytic spread reproduces the vconverge sweep within 15%."""
    dictValidation = fdictLoadBudget()["dictValidation"]
    assert 0.85 < dictValidation["agreement_ratio"] < 1.15, (
        "analytic propagation disagrees with the forward-model sweep; the "
        "informed-case prediction is not trustworthy")


def test_forward_model_partition_is_linear():
    """HARD GATE: the additive variance partition requires a near-linear model."""
    dictLevel1 = fdictLoadBudget()["dictLevel1ForwardModel"]
    assert dictLevel1["iTrials"] > 200
    assert dictLevel1["dTotalRSquared"] > 0.95, (
        "log-flux is not linear in the sampled inputs; the block-wise "
        "variance partition is not valid")


def test_variance_shares_are_physical():
    """Every first-order index lies in [0, 1] and the shares roughly close."""
    for sKey in ("dictLevel2PopulationZ", "dictLevel2InformedZ"):
        dictShares = fdictLoadBudget()[sKey]["dictSourceShares"]
        daShares = np.array([d["dVarianceShare"] for d in dictShares.values()])
        assert np.all(daShares >= 0.0) and np.all(daShares <= 1.0)
        assert 0.9 < daShares.sum() < 1.15


def test_measurement_narrows_the_flux():
    """The X-ray measurement must reduce, never inflate, the flux spread."""
    dictBudget = fdictLoadBudget()
    dPopulation = dictBudget["dictPopulationZ"]["sigma_log10"]
    dInformed = dictBudget["dictInformedZ"]["sigma_log10"]
    assert 0 < dInformed < dPopulation
    assert dictBudget["dSpreadReductionFactor"] > 1.0


def test_measurement_collapses_the_population_term():
    """The measurement must shrink the population-scatter term specifically."""
    dictBudget = fdictLoadBudget()
    dBefore = (dictBudget["dictLevel2PopulationZ"]["dictSourceShares"]
               [S_POPULATION_SOURCE]["dSpreadDex"])
    dAfter = (dictBudget["dictLevel2InformedZ"]["dictSourceShares"]
              [S_POPULATION_SOURCE]["dSpreadDex"])
    assert dAfter < dBefore
    dConversion = (dictBudget["dictLevel2InformedZ"]["dictSourceShares"]
                   [S_CONVERSION_SOURCE]["dSpreadDex"])
    assert dConversion > dAfter, (
        "with the measurement in hand, the band conversion should become the "
        "dominant remaining uncertainty")


def test_lognormal_mean_exceeds_median():
    """Both distributions are right-skewed: the mean must exceed the median."""
    for sKey in ("dictPopulationZ", "dictInformedZ"):
        dictStats = fdictLoadBudget()[sKey]
        assert dictStats["mean"] > dictStats["median"]
        assert dictStats["mean_over_median"] > 1.0
