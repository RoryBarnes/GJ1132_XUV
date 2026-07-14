"""Integrity checks for the cumulative-XUV variance budget.

Two gates are load-bearing. The analytic propagation must reproduce the
vconverge forward-model sweep in the population case, or none of the projected
states of knowledge can be trusted. And the level-1 decomposition must remain
near-linear (high R^2), or the additive partition of variance across input
blocks is not valid.
"""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_SCENARIOS = ["noMeasurement", "xrayMeasurement", "panchromaticSed",
                "relationFloor"]
S_POPULATION_SOURCE = "stellar population (intrinsic scatter)"
S_CONVERSION_SOURCE = "band conversion (intrinsic scatter)"


def fdictLoadBudget():
    """Load the variance-budget product."""
    with open(os.path.join(S_DIRECTORY,
                           "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def test_flux_samples_positive_and_finite():
    """Every scenario's flux distribution is populated, finite, and positive."""
    saFiles = [f"fluxSamples_{sKey}.txt" for sKey in SA_SCENARIOS]
    saFiles.append("fluxSamplesForwardModelSweep.txt")
    for sName in saFiles:
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sName))
        assert daFlux.size > 100
        assert np.all(np.isfinite(daFlux))
        assert np.all(daFlux > 0)


def test_analytic_propagation_matches_forward_model():
    """HARD GATE: analytic spread reproduces the vconverge sweep within 15%."""
    dictValidation = fdictLoadBudget()["dictValidation"]
    assert 0.85 < dictValidation["agreement_ratio"] < 1.15, (
        "analytic propagation disagrees with the forward-model sweep; the "
        "projected states of knowledge are not trustworthy")


def test_forward_model_partition_is_linear():
    """HARD GATE: the additive variance partition requires a near-linear model."""
    dictLevel1 = fdictLoadBudget()["dictLevel1ForwardModel"]
    assert dictLevel1["iTrials"] > 200
    assert dictLevel1["dTotalRSquared"] > 0.95, (
        "log-flux is not linear in the sampled inputs; the block-wise "
        "variance partition is not valid")


def test_variance_shares_are_physical():
    """Every first-order index lies in [0, 1] and the shares roughly close."""
    dictShares = fdictLoadBudget()["dictLevel2SourceIndices"]["dictSourceShares"]
    daShares = np.array([d["dVarianceShare"] for d in dictShares.values()])
    assert np.all(daShares >= 0.0) and np.all(daShares <= 1.0)
    assert 0.9 < daShares.sum() < 1.15


def test_scenarios_are_monotonically_tighter():
    """Each successive state of knowledge must narrow the flux, never widen it."""
    dictScenarios = fdictLoadBudget()["dictScenarios"]
    daSpreads = np.array([dictScenarios[sKey]["sigma_log10"]
                          for sKey in SA_SCENARIOS])
    assert np.all(daSpreads > 0)
    assert np.all(np.diff(daSpreads) < 0), (
        "knowing more about the star must not widen its inferred XUV history")


def test_stacked_variance_reconstructs_each_total():
    """Each scenario's per-source variances must sum to its total variance."""
    for sKey, dictScenario in fdictLoadBudget()["dictScenarios"].items():
        dTotal = dictScenario["sigma_log10"] ** 2
        dStacked = sum(dictScenario["dictVarianceBySource"].values())
        assert abs(dStacked - dTotal) < 0.02 * max(dTotal, 1e-3), (
            f"{sKey}: stacked variance segments do not reconstruct the total")


def test_measurement_moves_the_bottleneck_to_the_conversion():
    """With the X-ray in hand, band conversion must dominate the remainder."""
    dictVariance = (fdictLoadBudget()["dictScenarios"]["xrayMeasurement"]
                    ["dictVarianceBySource"])
    assert dictVariance[S_CONVERSION_SOURCE] > dictVariance[S_POPULATION_SOURCE]


def test_preposterior_satisfies_the_tower_property():
    """The future posterior's centre-spread and width must reconstruct today's.

    Var[now] = Var[future centre] + E[Var[future]]. If this fails, the fan of
    possible future posteriors would not average back to what we believe now,
    and the projection would be incoherent.
    """
    dictBudget = fdictLoadBudget()
    dictPre = dictBudget["dictPreposterior"]
    dNow = dictBudget["dictScenarios"][dictPre["sCurrentState"]]["sigma_log10"]
    dFuture = dictBudget["dictScenarios"][
        dictPre["sFutureState"]]["sigma_log10"]
    dReconstructed = np.sqrt(dictPre["dSigmaCenterDex"] ** 2 + dFuture ** 2)
    assert abs(dReconstructed - dNow) < 0.01 * dNow


def test_projected_medians_are_an_artifact_not_a_prediction():
    """Guard the figure's honesty: the projected centres are NOT predictions.

    Freezing sources at their means preserves today's central estimate, so the
    projected states share a median by construction. That is only defensible
    while the preposterior records that a real measurement could land elsewhere.
    """
    dictBudget = fdictLoadBudget()
    daMedians = np.array([dictBudget["dictScenarios"][sKey]["median"]
                          for sKey in ("xrayMeasurement", "panchromaticSed",
                                       "relationFloor")])
    assert np.ptp(daMedians) / np.mean(daMedians) < 0.05
    assert dictBudget["dictPreposterior"]["dSigmaCenterDex"] > 0.1, (
        "projected medians coincide with today's, so the preposterior MUST "
        "record that a future measurement could centre the answer elsewhere")


def test_shoreline_verdict_probabilities_are_coherent():
    """Straddling the shoreline must be likelier than falling below it."""
    dictPre = fdictLoadBudget()["dictPreposterior"]
    dStraddle = dictPre["dProbabilityStillStraddlesShoreline"]
    dBelow = dictPre["dProbabilityMedianBelowShoreline"]
    assert 0.0 <= dBelow <= dStraddle <= 1.0


def test_lognormal_mean_exceeds_median():
    """Every distribution is right-skewed: the mean must exceed the median."""
    for sKey, dictScenario in fdictLoadBudget()["dictScenarios"].items():
        assert dictScenario["mean"] > dictScenario["median"]
        assert dictScenario["mean_over_median"] > 1.0
