"""Integrity checks for the X-ray-measurement-value analysis.

The validation test is the load-bearing one: the analytic uncertainty
propagation must reproduce the vconverge forward-model sweep in the population
case, or the informed-case prediction it makes cannot be trusted.
"""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_SAMPLE_FILES = ["fluxSamplesPopulationZ.txt", "fluxSamplesInformedZ.txt",
                   "fluxSamplesForwardModelSweep.txt"]


def fdictLoadBudget():
    """Load the uncertainty-budget product."""
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
        "analytic uncertainty propagation disagrees with the forward-model "
        "sweep; the informed-case prediction is not trustworthy")


def test_measurement_narrows_the_flux():
    """The X-ray measurement must reduce, never inflate, the flux spread."""
    dictBudget = fdictLoadBudget()
    dSpreadPopulation = dictBudget["dictPopulationZ"]["sigma_log10"]
    dSpreadInformed = dictBudget["dictInformedZ"]["sigma_log10"]
    assert 0 < dSpreadInformed < dSpreadPopulation
    assert dictBudget["dSpreadReductionFactor"] > 1.0


def test_population_scatter_dominates_without_the_measurement():
    """Without the measurement, population scatter is the largest component."""
    dictComponents = fdictLoadBudget()["dictComponentsDex"]
    dPopulation = dictComponents["D_population_scatter_population_z"]
    for sKey in ("A_refit_mean_line", "B_conversion_covariance",
                 "C_conversion_scatter"):
        assert dictComponents[sKey] < dPopulation


def test_lognormal_mean_exceeds_median():
    """Both distributions are right-skewed: the mean must exceed the median."""
    for sKey in ("dictPopulationZ", "dictInformedZ"):
        dictStats = fdictLoadBudget()[sKey]
        assert dictStats["mean"] > dictStats["median"]
        assert dictStats["mean_over_median"] > 1.0
