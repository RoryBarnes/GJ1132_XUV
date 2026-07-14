#!/usr/bin/env python3
"""
Reconstruct WHEN GJ 1132 b accumulated its cumulative XUV flux.

Each vconverge trial in the Engle sweep wrote a forward trajectory of
cumulative XUV flux versus time. Stacking those trajectories on a common
stellar-age grid gives the accumulation history: the median cumulative flux
reached by every age, with its 68% envelope. The flux is dominated by the
early saturated phase, so the history climbs steeply and then flattens long
before the present day -- by the lower bound of the age posterior the planet
has already received nearly all of its lifetime dose.

The step also quantifies that flatness two ways, which must agree:

  - by integration: the fraction of the final flux already in place at each age;
  - by the local derivative: the elasticity d ln(F) / d ln(age) near the
    present, which times the age-posterior width gives age's contribution to
    the flux spread (independently reproducing the ~0.01% variance share the
    forward-model regression assigns to the present age).

Trajectories are included at a given age only while they extend that far (each
trial stops at its own sampled present age), and the median is reported with
its trajectory count so thinning coverage at old ages is explicit.

Reference: Saltelli et al. (2008) for the elasticity / sensitivity framing.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (D_CUMULATIVE_EARTH_FLUX, D_SHORELINE_FLUX,
                                 D_LOWER_BOUND, D_UPPER_BOUND)

I_GRID_POINTS = 240
I_MIN_COVERAGE = 100
D_COVERAGE_FRACTION = 0.3
I_SHOWN_TRAJECTORIES = 25
I_SHOWN_RESAMPLE = 160
DA_MILESTONE_AGES = np.array([0.5, 1.0, 2.0, 3.0, 5.0])


def fdReadOption(sPath, sKey):
    """Return one option value from a vplanet input file."""
    matchOption = re.search(rf"^{sKey}\s+(\S+)", open(sPath).read(), re.M)
    return float(matchOption.group(1)) if matchOption else np.nan


def ftLoadTrajectory(sTrialDirectory):
    """Return (age [Gyr], cumulative flux [Earth units], saturation age [Gyr]).

    The saturation age is 10^d, where d = dXUVEngleMidLateD is the break of the
    sampled XUV relation, expressed in log10(age/Gyr): activity is saturated
    below it and declines above it.
    """
    saForward = glob.glob(os.path.join(sTrialDirectory, "*.b.forward"))
    if not saForward:
        return None
    daForward = np.loadtxt(saForward[0])
    if daForward.ndim != 2 or len(daForward) < 5:
        return None
    sStarPath = os.path.join(sTrialDirectory, "star.in")
    dStartAge = fdReadOption(sStarPath, "dAge")
    dBreak = fdReadOption(sStarPath, "dXUVEngleMidLateD")
    if not np.isfinite(dStartAge) or not np.isfinite(dBreak):
        return None
    daAgeGyr = (dStartAge + daForward[:, 0]) / 1e9
    return daAgeGyr, daForward[:, 1] / D_CUMULATIVE_EARTH_FLUX, 10.0 ** dBreak


def flistLoadTrajectories(sSweepDirectory):
    """Load usable flux trajectories, applying the pipeline's flux filter.

    The same final-flux window [D_LOWER_BOUND, D_UPPER_BOUND] that
    daExtractFluxValues applies everywhere else is applied here, so this step
    summarizes the identical population as the flux-distribution steps rather
    than including unconverged extremes.
    """
    listTrajectories, listSaturationAges = [], []
    for sTrial in sorted(glob.glob(os.path.join(sSweepDirectory,
                                                "*_xuv_rand_*"))):
        tTrajectory = ftLoadTrajectory(sTrial)
        if tTrajectory is not None and \
                D_LOWER_BOUND <= tTrajectory[1][-1] <= D_UPPER_BOUND:
            listTrajectories.append((tTrajectory[0], tTrajectory[1]))
            listSaturationAges.append(tTrajectory[2])
    if len(listTrajectories) < I_MIN_COVERAGE:
        raise ValueError(f"only {len(listTrajectories)} usable trajectories "
                         f"in {sSweepDirectory}")
    return listTrajectories, np.array(listSaturationAges)


def fdMaxCoveredAge(listTrajectories, iFloor):
    """Return the oldest age still reached by at least iFloor trajectories."""
    daPresentAge = np.sort([daAge[-1] for daAge, _ in listTrajectories])
    return float(daPresentAge[-iFloor])


def fdictStackOnAgeGrid(listTrajectories, dMaxAge):
    """Return per-age percentiles over a fixed cohort of complete trajectories.

    The envelope is taken over the trajectories that reach the oldest
    well-covered age, so every age is summarized over the SAME stars. This
    avoids a shifting-subset artifact -- percentiles taken over trajectories
    that merely extend past each age would not be monotone as coverage thins.
    The age posterior's interval is drawn independently and can extend past
    this covered curve, where too few stars are that old to summarize.
    """
    iFloor = max(I_MIN_COVERAGE,
                 int(D_COVERAGE_FRACTION * len(listTrajectories)))
    dCoveredAge = min(dMaxAge, fdMaxCoveredAge(listTrajectories, iFloor))
    listCohort = [(daAge, daFlux) for daAge, daFlux in listTrajectories
                  if daAge[-1] >= dCoveredAge]
    daAgeGrid = np.linspace(0.0, dCoveredAge, I_GRID_POINTS)
    daStack = np.array([[np.interp(dAge, daAge, daFlux) for dAge in daAgeGrid]
                        for daAge, daFlux in listCohort])
    return {
        "daAgeGrid": daAgeGrid.tolist(),
        "daMedianFlux": np.median(daStack, axis=0).tolist(),
        "daLowerFlux": np.percentile(daStack, 16, axis=0).tolist(),
        "daUpperFlux": np.percentile(daStack, 84, axis=0).tolist(),
        "iCohort": len(listCohort),
    }


def flistSelectShownTrajectories(listTrajectories):
    """Return a reproducible, spread-spanning set of real trajectories to plot.

    Trajectories are ranked by final cumulative flux and sampled at evenly
    spaced quantiles -- a deterministic rule (no RNG) that reproduces the same
    set from the same sweep and spans the flux distribution rather than
    over-showing its bulk. Each is resampled onto its own age range so the
    stored curves stay compact while each still ends at its trial's real
    present age, giving a genuine (present age, total dose) endpoint.
    """
    listSorted = sorted(listTrajectories, key=lambda t: t[1][-1])
    daRanks = np.linspace(0, len(listSorted) - 1, I_SHOWN_TRAJECTORIES)
    listShown = []
    for dRank in daRanks:
        daAge, daFlux = listSorted[int(round(dRank))]
        daGrid = np.linspace(daAge[0], daAge[-1], I_SHOWN_RESAMPLE)
        listShown.append({"daAgeGrid": daGrid.tolist(),
                          "daFlux": np.interp(daGrid, daAge, daFlux).tolist(),
                          "dPresentAge": float(daAge[-1]),
                          "dFinalFlux": float(daFlux[-1])})
    return listShown


def fdMeanFractionAtAge(listTrajectories, dAge):
    """Mean over trials of the fraction of each trial's OWN final flux at dAge.

    Computed per trajectory, so it is a true fraction bounded by 1 regardless
    of how coverage thins with age -- unlike a ratio of the stacked medians,
    whose numerator and denominator are drawn from different trial subsets.
    """
    listFraction = [np.interp(dAge, daAge, daFlux) / daFlux[-1]
                    for daAge, daFlux in listTrajectories if daAge[-1] >= dAge]
    return float(np.mean(listFraction))


def fdictFractionMilestones(listTrajectories):
    """Fraction of each trial's FINAL flux already in place at milestone ages."""
    return {f"{dAge:.1f}": fdMeanFractionAtAge(listTrajectories, dAge)
            for dAge in DA_MILESTONE_AGES}


def fdComputeElasticity(listTrajectories):
    """Return the mean local elasticity d ln(F) / d ln(age) near the present."""
    listElasticity = []
    for daAge, daFlux in listTrajectories:
        dEnd = daAge[-1]
        if dEnd < 4.0:
            continue
        dFluxEnd, dFluxBack = daFlux[-1], np.interp(0.9 * dEnd, daAge, daFlux)
        listElasticity.append((np.log(dFluxEnd) - np.log(dFluxBack))
                              / (np.log(dEnd) - np.log(0.9 * dEnd)))
    return float(np.mean(listElasticity))


def fdictSummarizeAge(daAgeSamples):
    """Return the age posterior's median and 95% interval in Gyr."""
    daAgeGyr = daAgeSamples / 1e9
    return {
        "median": float(np.median(daAgeGyr)),
        "ci95": [float(np.percentile(daAgeGyr, 2.5)),
                 float(np.percentile(daAgeGyr, 97.5))],
        "sigma_log10": float(np.std(np.log10(daAgeGyr))),
    }


def fdictSummarizeSaturation(daSaturationAges):
    """Return the saturation-age median and 95% interval in Gyr."""
    return {
        "median": float(np.median(daSaturationAges)),
        "ci95": [float(np.percentile(daSaturationAges, 2.5)),
                 float(np.percentile(daSaturationAges, 97.5))],
    }


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reconstruct when GJ 1132 b accumulated its XUV flux.")
    parser.add_argument("--converged-flux", required=True,
                        help="Engle Cumulative XUV output; its directory holds "
                             "the per-trial flux trajectories.")
    parser.add_argument("--age-samples", required=True,
                        help="Rotation-only age samples (.txt, years).")
    return parser.parse_args()


def main():
    """Build the accumulation history and its flatness diagnostics."""
    args = ftParseArguments()
    listTrajectories, daSaturationAges = flistLoadTrajectories(
        os.path.dirname(args.converged_flux))
    dictAge = fdictSummarizeAge(np.loadtxt(args.age_samples))
    dictHistory = fdictStackOnAgeGrid(listTrajectories, dictAge["ci95"][1])
    dElasticity = fdComputeElasticity(listTrajectories)
    dictSummary = {
        "dictAgePosterior": dictAge,
        "dictSaturationAge": fdictSummarizeSaturation(daSaturationAges),
        "dictHistory": dictHistory,
        "listShownTrajectories": flistSelectShownTrajectories(listTrajectories),
        "dictFractionMilestones": fdictFractionMilestones(listTrajectories),
        "dFractionByAgeLowerBound": fdMeanFractionAtAge(
            listTrajectories, dictAge["ci95"][0]),
        "dElasticityNearPresent": dElasticity,
        "dAgeContributionDex": dElasticity * dictAge["sigma_log10"],
        "dShorelineEarthFlux": D_SHORELINE_FLUX,
        "iTrajectories": len(listTrajectories),
    }
    with open("accumulationHistory.json", "w") as fileHandle:
        json.dump(dictSummary, fileHandle, indent=2)
    fnPrintSummary(dictSummary)


def fnPrintSummary(dictSummary):
    """Print the accumulation milestones and the flatness diagnostics."""
    print(f"Reconstructed from {dictSummary['iTrajectories']} flux "
          f"trajectories.")
    print("Fraction of final cumulative XUV flux accumulated by age:")
    for sAge, dFraction in dictSummary["dictFractionMilestones"].items():
        print(f"  by {sAge} Gyr: {100 * dFraction:5.1f}%")
    dictSat = dictSummary["dictSaturationAge"]
    print(f"Saturation age: median {dictSat['median']:.2f} Gyr, 95% CI "
          f"[{dictSat['ci95'][0]:.2f}, {dictSat['ci95'][1]:.2f}] Gyr "
          f"(activity declines after this).")
    dictAge = dictSummary["dictAgePosterior"]
    print(f"Age 95% CI [{dictAge['ci95'][0]:.2f}, {dictAge['ci95'][1]:.2f}] "
          f"Gyr; by its LOWER bound the planet already has "
          f"{100 * dictSummary['dFractionByAgeLowerBound']:.1f}% of its dose.")
    print(f"Elasticity d ln(F)/d ln(age) near present = "
          f"{dictSummary['dElasticityNearPresent']:.3f}  ->  age contributes "
          f"{dictSummary['dAgeContributionDex']:.4f} dex to the flux spread.")


if __name__ == "__main__":
    main()
