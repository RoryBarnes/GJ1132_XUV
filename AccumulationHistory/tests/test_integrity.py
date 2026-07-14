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


def test_shown_trajectories_are_real_and_span_the_range():
    """The 25 plotted simulations are complete runs spanning the flux spread."""
    listShown = fdictLoadHistory()["listShownTrajectories"]
    assert len(listShown) == 25
    for dictTrajectory in listShown:
        daFlux = np.array(dictTrajectory["daFlux"])
        daAge = np.array(dictTrajectory["daAgeGrid"])
        assert np.all(np.diff(daFlux) >= -1e-9)
        assert np.all(np.diff(daAge) > 0)
        assert abs(daFlux[-1] - dictTrajectory["dFinalFlux"]) < 1e-6
        assert abs(daAge[-1] - dictTrajectory["dPresentAge"]) < 1e-6
    daFinal = np.array([d["dFinalFlux"] for d in listShown])
    assert daFinal[-1] > 3 * daFinal[0], "shown set should span the spread"


def test_shown_endpoints_sample_the_age_interval():
    """The present-day endpoints fall around the age posterior's interval."""
    dictSummary = fdictLoadHistory()
    daEndAge = np.array([d["dPresentAge"]
                         for d in dictSummary["listShownTrajectories"]])
    dLower, dUpper = dictSummary["dictAgePosterior"]["ci95"]
    dMedian = dictSummary["dictAgePosterior"]["median"]
    assert dLower - 1 < np.median(daEndAge) < dUpper + 1
    assert abs(np.median(daEndAge) - dMedian) < 1.0


def test_main_sequence_arrival_is_tight_and_early():
    """The main-sequence arrival is a tight, early epoch after artifact cuts.

    The luminosity-minimum estimator is bimodal (a Baraffe grid artifact puts
    ~13% of tracks near 0.5 Gyr); after the data-driven cut the physical
    cluster must be tight, as stellar age varies only a few percent across the
    mass posterior.
    """
    dictMs = fdictLoadHistory()["dictMainSequenceArrival"]
    assert 1.0 < dictMs["mean"] < 2.5
    assert dictMs["fractional_spread"] < 0.12, (
        "the physical main-sequence-arrival cluster should be tight")
    assert dictMs["fArtifactFraction"] < 0.3


def test_causal_chain_is_ordered():
    """PMS ends, then saturation ends, then the star reaches its present age."""
    dictSummary = fdictLoadHistory()
    dMs = dictSummary["dictMainSequenceArrival"]["mean"]
    dSatUpper = dictSummary["dictSaturationAge"]["ci95"][1]
    dAgeLower = dictSummary["dictAgePosterior"]["ci95"][0]
    assert dMs < dSatUpper < dAgeLower, (
        "the physical ordering main-sequence -> saturation end -> present age "
        "must hold")


def test_saturation_precedes_and_shapes_the_plateau():
    """Saturation ends early and well before the stellar-age interval."""
    dictSummary = fdictLoadHistory()
    dictSat = dictSummary["dictSaturationAge"]
    dLower, dMedian, dUpper = (dictSat["ci95"][0], dictSat["median"],
                               dictSat["ci95"][1])
    assert 0 < dLower < dMedian < dUpper
    assert dUpper < dictSummary["dictAgePosterior"]["ci95"][0], (
        "the saturation phase must end before the youngest allowed stellar age")
    dMostAccumulated = dictSummary["dictFractionMilestones"]["3.0"]
    assert dMostAccumulated > 0.5, (
        "most of the dose should arrive around the saturation epoch")


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
