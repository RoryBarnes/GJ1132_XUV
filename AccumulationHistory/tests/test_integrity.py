"""Integrity checks for the cumulative-XUV accumulation history."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fdictLoadHistory():
    """Load the accumulation-history product."""
    with open(os.path.join(S_DIRECTORY,
                           "accumulationHistory.json")) as fileHandle:
        return json.load(fileHandle)


def test_history_is_monotone_and_bracketed():
    """The median accumulation curve rises monotonically inside its envelope."""
    dictHistory = fdictLoadHistory()["dictHistory"]
    daMedian = np.array(dictHistory["daMedianFlux"])
    daLower = np.array(dictHistory["daLowerFlux"])
    daUpper = np.array(dictHistory["daUpperFlux"])
    assert daMedian.size > 50
    assert np.all(np.diff(daMedian) >= -1e-6 * daMedian[-1])
    assert np.all(daLower <= daMedian) and np.all(daMedian <= daUpper)


def test_fraction_milestones_increase_and_saturate():
    """The accumulated fraction rises with age and most arrives early."""
    dictMilestones = fdictLoadHistory()["dictFractionMilestones"]
    daAges = np.array(sorted(float(s) for s in dictMilestones))
    daFraction = np.array([dictMilestones[f"{dAge:.1f}"] for dAge in daAges])
    assert np.all(np.diff(daFraction) > 0)
    assert np.all((daFraction > 0) & (daFraction <= 1.0))
    assert dictMilestones["3.0"] > 0.5, "most of the dose should arrive early"


def test_dose_essentially_complete_by_age_lower_bound():
    """By the lower edge of the age interval the dose is nearly all delivered."""
    dictSummary = fdictLoadHistory()
    dFraction = dictSummary["dFractionByAgeLowerBound"]
    assert 0.9 < dFraction <= 1.0, (
        "the planet should have almost its entire dose by the youngest age "
        "the posterior allows")


def test_age_contribution_is_negligible():
    """Elasticity times the age width gives a tiny flux contribution."""
    dictSummary = fdictLoadHistory()
    assert 0 < dictSummary["dElasticityNearPresent"] < 1.0
    assert dictSummary["dAgeContributionDex"] < 0.05, (
        "the present age must contribute little to the cumulative-flux spread")


def test_age_interval_lies_on_the_plateau():
    """The age 95% interval sits where the accumulation curve has flattened."""
    dictSummary = fdictLoadHistory()
    dictHistory = dictSummary["dictHistory"]
    daAge = np.array(dictHistory["daAgeGrid"])
    daMedian = np.array(dictHistory["daMedianFlux"])
    dLowerAge = dictSummary["dictAgePosterior"]["ci95"][0]
    if dLowerAge > daAge[-1]:
        return
    dSlopeEnd = ((daMedian[-1] - np.interp(dLowerAge, daAge, daMedian))
                 / daMedian[-1])
    assert abs(dSlopeEnd) < 0.1, (
        "flux should change by <10% across the covered age interval")
