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
from utils.cumulativeXuv import D_CUMULATIVE_EARTH_FLUX, D_SHORELINE_FLUX

I_GRID_POINTS = 240
I_MIN_COVERAGE = 100
D_COVERAGE_FRACTION = 0.3
DA_MILESTONE_AGES = np.array([0.5, 1.0, 2.0, 3.0, 5.0])


def fdReadOption(sPath, sKey):
    """Return one option value from a vplanet input file."""
    matchOption = re.search(rf"^{sKey}\s+(\S+)", open(sPath).read(), re.M)
    return float(matchOption.group(1)) if matchOption else np.nan


def ftLoadTrajectory(sTrialDirectory):
    """Return (absolute stellar age [Gyr], cumulative flux [Earth units])."""
    saForward = glob.glob(os.path.join(sTrialDirectory, "*.b.forward"))
    if not saForward:
        return None
    daForward = np.loadtxt(saForward[0])
    if daForward.ndim != 2 or len(daForward) < 5:
        return None
    dStartAge = fdReadOption(os.path.join(sTrialDirectory, "star.in"), "dAge")
    if not np.isfinite(dStartAge):
        return None
    daAgeGyr = (dStartAge + daForward[:, 0]) / 1e9
    return daAgeGyr, daForward[:, 1] / D_CUMULATIVE_EARTH_FLUX


def flistLoadTrajectories(sSweepDirectory):
    """Load every usable flux trajectory from the sweep."""
    listTrajectories = []
    for sTrial in sorted(glob.glob(os.path.join(sSweepDirectory,
                                                "*_xuv_rand_*"))):
        tTrajectory = ftLoadTrajectory(sTrial)
        if tTrajectory is not None and tTrajectory[1][-1] > 0:
            listTrajectories.append(tTrajectory)
    if len(listTrajectories) < I_MIN_COVERAGE:
        raise ValueError(f"only {len(listTrajectories)} usable trajectories "
                         f"in {sSweepDirectory}")
    return listTrajectories


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
    listTrajectories = flistLoadTrajectories(
        os.path.dirname(args.converged_flux))
    dictAge = fdictSummarizeAge(np.loadtxt(args.age_samples))
    dictHistory = fdictStackOnAgeGrid(listTrajectories, dictAge["ci95"][1])
    dElasticity = fdComputeElasticity(listTrajectories)
    dictSummary = {
        "dictAgePosterior": dictAge,
        "dictHistory": dictHistory,
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
    dictAge = dictSummary["dictAgePosterior"]
    print(f"Age 95% CI [{dictAge['ci95'][0]:.2f}, {dictAge['ci95'][1]:.2f}] "
          f"Gyr; by its LOWER bound the planet already has "
          f"{100 * dictSummary['dFractionByAgeLowerBound']:.1f}% of its dose.")
    print(f"Elasticity d ln(F)/d ln(age) near present = "
          f"{dictSummary['dElasticityNearPresent']:.3f}  ->  age contributes "
          f"{dictSummary['dAgeContributionDex']:.4f} dex to the flux spread.")


if __name__ == "__main__":
    main()
