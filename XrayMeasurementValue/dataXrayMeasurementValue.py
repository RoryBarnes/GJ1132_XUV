#!/usr/bin/env python3
"""
Variance budget for GJ 1132 b's cumulative XUV flux, and the value of an X-ray.

The budget is computed at two nested levels, both variance-based.

LEVEL 1 -- forward model, all sampled inputs. Every vconverge trial directory
records the inputs it was run with (star.in, b.in, vpl.in) and the cumulative
flux it produced, so the sweep IS a design matrix. Regressing log10(F) on the
standardized inputs gives the variance share of each input block. The blocks
are mutually independent by construction (vspace samples them independently),
so their contributions partition the variance additively; the linear model's
R^2 is reported as the validity check on that partition.

LEVEL 2 -- inside the block that dominates. The four X-UV coefficients carry
four independent sources of uncertainty, which the prior table folds together:
the refit posterior covariance, the band-conversion coefficient covariance, the
band-conversion intrinsic scatter, and the star's offset z within the
population's intrinsic scatter. Shares are FIRST-ORDER VARIANCE INDICES
computed by ablation -- freeze one source at its mean, remeasure the variance,
and attribute the drop -- rather than a quadrature sum, because the conversion
slope multiplies several terms and they are therefore not independent.

The same machinery then propagates two states of knowledge that differ ONLY in
z: the population prior (no host-star X-ray measurement) and the L_X-informed
posterior. The age prior is identical in both, so the difference between the
resulting flux distributions is the value of the measurement.

Because the flux scales as 10^offset, the propagated distribution is lognormal:
its MEDIAN tracks the central relation while its MEAN is inflated by
exp(sigma_ln^2/2). Reporting the mean alone conflates a wider error bar with a
larger flux.

References: Saltelli et al. (2008), Global Sensitivity Analysis: The Primer
(variance-based decomposition and standardized regression); Kelly (2007) ApJ
665, 1489 (measurement-error framework).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (fdictLoadConvergedJson, daExtractFluxValues,
                                 D_SHORELINE_FLUX)

I_SEED = 42
I_NUM_SAMPLES = 200000
D_TAU_GJ1132 = 0.8533

SA_STAR_KEYS = ["dMass", "dAge", "dXUVEngleMidLateA", "dXUVEngleMidLateB",
                "dXUVEngleMidLateC", "dXUVEngleMidLateD"]
SA_PLANET_KEYS = ["dMass", "dRadius", "dEcc", "dOrbPeriod"]
SA_INPUT_NAMES = (["star_dMass", "star_dAge", "engle_a", "engle_b", "engle_c",
                   "engle_d"] + [f"planet_{s}" for s in SA_PLANET_KEYS]
                  + ["dStopTime"])
DICT_BLOCKS = {
    "X-UV relation coefficients": [2, 3, 4, 5],
    "planet parameters": [6, 7, 8, 9],
    "stellar mass": [0],
    "stellar age at present": [10],
    "stellar age at start": [1],
}


def fdReadOption(sPath, sKey):
    """Return one option value from a vplanet input file."""
    matchOption = re.search(rf"^{sKey}\s+(\S+)", open(sPath).read(), re.M)
    return float(matchOption.group(1)) if matchOption else np.nan


def fdFinalFlux(sTrialDirectory):
    """Return the trial's final cumulative XUV flux, or NaN if unreadable."""
    saLogs = glob.glob(os.path.join(sTrialDirectory, "*.log"))
    if not saLogs:
        return np.nan
    sFinal = open(saLogs[0]).read().split("FINAL SYSTEM PROPERTIES")[-1]
    matchFlux = re.search(r"\(CumulativeXUVFlux\)[^:]*: (\S+)", sFinal)
    return float(matchFlux.group(1)) if matchFlux else np.nan


def fdaTrialInputs(sTrialDirectory):
    """Return the sampled input vector recorded in one trial directory."""
    daStar = [fdReadOption(os.path.join(sTrialDirectory, "star.in"), sKey)
              for sKey in SA_STAR_KEYS]
    daPlanet = [fdReadOption(os.path.join(sTrialDirectory, "b.in"), sKey)
                for sKey in SA_PLANET_KEYS]
    dStopTime = fdReadOption(os.path.join(sTrialDirectory, "vpl.in"),
                             "dStopTime")
    return np.array(daStar + daPlanet + [dStopTime])


def ftBuildDesignMatrix(sSweepDirectory):
    """Assemble (inputs, log10 flux) over every trial the sweep recorded."""
    listInputs, listFlux = [], []
    for sTrial in sorted(glob.glob(os.path.join(sSweepDirectory,
                                                "*_xuv_rand_*"))):
        dFlux = fdFinalFlux(sTrial)
        daInputs = fdaTrialInputs(sTrial)
        if dFlux > 0 and np.all(np.isfinite(daInputs)):
            listInputs.append(daInputs)
            listFlux.append(dFlux)
    if len(listInputs) < 50:
        raise ValueError(f"only {len(listInputs)} usable trials in "
                         f"{sSweepDirectory}; cannot decompose variance")
    return np.array(listInputs), np.log10(np.array(listFlux))


def fdExplainedVariance(daPredictors, daResponse):
    """Return the R^2 of an ordinary least-squares fit."""
    daDesign = np.column_stack([np.ones(len(daPredictors)), daPredictors])
    daCoefficients, _, _, _ = np.linalg.lstsq(daDesign, daResponse, rcond=None)
    daResidual = daResponse - daDesign @ daCoefficients
    return float(1.0 - np.var(daResidual) / np.var(daResponse))


def fdictDecomposeForwardModel(sSweepDirectory):
    """Level 1: variance share of each independent input block in the sweep."""
    daInputs, daLogFlux = ftBuildDesignMatrix(sSweepDirectory)
    daStandard = (daInputs - daInputs.mean(0)) / daInputs.std(0)
    dictShares = {sBlock: fdExplainedVariance(daStandard[:, listIndices],
                                              daLogFlux)
                  for sBlock, listIndices in DICT_BLOCKS.items()}
    return {
        "iTrials": int(len(daLogFlux)),
        "dTotalRSquared": fdExplainedVariance(daStandard, daLogFlux),
        "dictBlockShares": {k: float(v) for k, v in dictShares.items()},
        "dSigmaLog10Flux": float(np.std(daLogFlux)),
        "sMethod": "standardized-regression variance decomposition over "
                   "mutually independent input blocks (Saltelli et al. 2008); "
                   "R^2 validates the linear partition",
    }


def fdaSampleMeanLine(daRows, dTau):
    """The refit hinge evaluated at dTau, in the native X-ray band."""
    return (daRows[:, 6] * dTau + daRows[:, 7]
            + daRows[:, 8] * np.clip(dTau - daRows[:, 9], 0.0, None))


def fdaSampleScatterLaw(daRows, dTau, dictSummary):
    """Each posterior row's native-band sigma_int evaluated at dTau."""
    return np.exp(daRows[:, 10] + daRows[:, 11]
                  * (dTau - dictSummary["dPivotTau"])
                  / dictSummary["dScaleTau"])


def fdaDrawZ(sMode, daInformed, iCount):
    """Draw z from the population prior or the L_X-informed posterior."""
    if sMode == "population":
        return np.random.normal(0.0, 1.0, iCount)
    return daInformed[np.random.randint(0, len(daInformed), iCount)]


def fdictSampleSources(daChain, dictSummary, dictConversion, daInformed,
                       sMode):
    """Draw every uncertainty source once, so any subset can be frozen."""
    daRows = daChain[np.random.randint(0, len(daChain), I_NUM_SAMPLES)]
    daDraws = np.random.multivariate_normal(
        [dictConversion["slope"], dictConversion["intercept"]],
        np.array(dictConversion["covariance_slope_intercept"]), I_NUM_SAMPLES)
    dScatter = dictConversion["intrinsic_scatter_dex"][
        "fScatterPosteriorMedian"]
    return {
        "daNative": fdaSampleMeanLine(daRows, D_TAU_GJ1132),
        "daSigmaInt": fdaSampleScatterLaw(daRows, D_TAU_GJ1132, dictSummary),
        "daZ": fdaDrawZ(sMode, daInformed, I_NUM_SAMPLES),
        "daSlope": daDraws[:, 0],
        "daIntercept": daDraws[:, 1],
        "daConversionScatter": np.random.normal(0, dScatter, I_NUM_SAMPLES),
    }


def fdaComposeOffset(dictSources, saFrozen=()):
    """Compose the X-UV log-offset, freezing the named sources at their means."""
    dictUse = {sKey: (np.full_like(daValue, float(np.mean(daValue)))
                      if sKey in saFrozen else daValue)
               for sKey, daValue in dictSources.items()}
    return (dictUse["daSlope"]
            * (dictUse["daNative"] + dictUse["daZ"] * dictUse["daSigmaInt"])
            + dictUse["daIntercept"] + dictUse["daConversionScatter"])


DICT_SOURCE_ABLATIONS = {
    "X-UV relation coefficients (posterior covariance)": ("daNative",),
    "band conversion (coefficient covariance)": ("daSlope", "daIntercept"),
    "band conversion (intrinsic scatter)": ("daConversionScatter",),
    "stellar population (intrinsic scatter)": ("daZ", "daSigmaInt"),
}

# States of knowledge. Each freezes a subset of sources at its mean, so every
# scenario's spread is measured with the same estimator on the same draws and
# the four are strictly comparable. The two projected scenarios assume the
# measured offsets PERSIST over the star's history -- the flux is set by the
# saturated phase, which ended ~5 Gyr ago and cannot be observed.
SA_CONVERSION_SOURCES = ("daConversionScatter", "daSlope", "daIntercept")
DICT_SCENARIOS = {
    "noMeasurement": {
        "sLabel": "no measurement",
        "sZMode": "population", "saFrozen": (), "bProjected": False},
    "xrayMeasurement": {
        "sLabel": "X-ray measurement",
        "sZMode": "informed", "saFrozen": (), "bProjected": False},
    "panchromaticSed": {
        "sLabel": "panchromatic X-UV SED",
        "sZMode": "informed", "saFrozen": SA_CONVERSION_SOURCES,
        "bProjected": True},
    "relationFloor": {
        "sLabel": "relation floor (star known exactly)",
        "sZMode": "informed",
        "saFrozen": SA_CONVERSION_SOURCES + ("daZ", "daSigmaInt"),
        "bProjected": True},
}


def fdictDecomposeSources(dictSources):
    """Level 2: first-order variance indices by ablation of each source."""
    dTotal = float(np.var(fdaComposeOffset(dictSources)))
    dictShares = {}
    for sName, saFrozen in DICT_SOURCE_ABLATIONS.items():
        dFrozen = float(np.var(fdaComposeOffset(dictSources, saFrozen)))
        dictShares[sName] = {
            "dVarianceShare": (dTotal - dFrozen) / dTotal,
            "dSpreadDex": float(np.sqrt(max(dTotal - dFrozen, 0.0))),
        }
    return {
        "dTotalVariance": dTotal,
        "dTotalSpreadDex": float(np.sqrt(dTotal)),
        "dictSourceShares": dictShares,
        "sMethod": "first-order variance indices by ablation (freeze one "
                   "source at its mean, remeasure the variance); not a "
                   "quadrature sum, because the conversion slope multiplies "
                   "several sources",
    }


def fdictEvaluateScenario(dictSources, dictScenario, daReferenceOffset,
                          dSweepMedian):
    """Compose one state of knowledge: its offsets, flux, spread, and budget."""
    daOffset = fdaComposeOffset(dictSources, dictScenario["saFrozen"])
    daFlux = fdaAnchorFlux(daOffset, daReferenceOffset, dSweepMedian)
    dictSummary = fdictSummarizeFlux(daFlux, dictScenario["sLabel"])
    dictSummary["bProjected"] = dictScenario["bProjected"]
    dictSummary["saFrozenSources"] = list(dictScenario["saFrozen"])
    dictSummary["dictVarianceBySource"] = fdictVarianceBySource(
        dictSources, dictScenario["saFrozen"])
    return daFlux, dictSummary


def fdictVarianceBySource(dictSources, saFrozen):
    """Variance (dex^2) each source still contributes under one scenario."""
    dTotal = float(np.var(fdaComposeOffset(dictSources, saFrozen)))
    dictVariance = {}
    for sName, saSources in DICT_SOURCE_ABLATIONS.items():
        saBoth = tuple(saFrozen) + tuple(saSources)
        dRemaining = float(np.var(fdaComposeOffset(dictSources, saBoth)))
        dictVariance[sName] = max(dTotal - dRemaining, 0.0)
    dictVariance["interaction residual"] = max(
        dTotal - sum(dictVariance.values()), 0.0)
    return dictVariance


def fdictSummarizeFlux(daFlux, sLabel):
    """Lognormal-aware summary statistics for a flux distribution."""
    return {
        "label": sLabel,
        "median": float(np.median(daFlux)),
        "mean": float(np.mean(daFlux)),
        "ci95": [float(np.percentile(daFlux, 2.5)),
                 float(np.percentile(daFlux, 97.5))],
        "sigma_log10": float(np.std(np.log10(daFlux))),
        "mean_over_median": float(np.mean(daFlux) / np.median(daFlux)),
        "fraction_above_shoreline": float(np.mean(daFlux > D_SHORELINE_FLUX)),
    }


def fdaAnchorFlux(daOffset, daReferenceOffset, dSweepMedian):
    """Convert log-offsets to flux, anchored on the sweep's population median."""
    return dSweepMedian * 10.0 ** (daOffset - np.median(daReferenceOffset))


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Variance budget and the value of one host-star X-ray "
                    "measurement for GJ 1132 b's cumulative XUV flux.")
    parser.add_argument("--joint-chain", required=True)
    parser.add_argument("--fit-summary", required=True)
    parser.add_argument("--conversion-fit", required=True)
    parser.add_argument("--z-offsets", required=True,
                        help="L_X-informed z posterior samples (.txt).")
    parser.add_argument("--converged-flux", required=True,
                        help="Population-z vconverge output; its directory "
                             "also holds the per-trial design matrix.")
    return parser.parse_args()


def main():
    """Compute both levels of the budget and both flux distributions."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daChain = np.load(args.joint_chain)
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    with open(args.conversion_fit) as fileHandle:
        dictConversion = json.load(
            fileHandle)["primary_fit_all_targets_rosat_band"]
    daInformed = np.loadtxt(args.z_offsets)
    daSweepFlux = daExtractFluxValues(fdictLoadConvergedJson(
        args.converged_flux))

    dictSourcesByMode = {
        sMode: fdictSampleSources(daChain, dictSummary, dictConversion,
                                  daInformed, sMode)
        for sMode in ("population", "informed")}
    dSweepMedian = float(np.median(daSweepFlux))
    daReferenceOffset = fdaComposeOffset(dictSourcesByMode["population"])

    dictScenarios = {}
    for sKey, dictScenario in DICT_SCENARIOS.items():
        daFlux, dictScenarioSummary = fdictEvaluateScenario(
            dictSourcesByMode[dictScenario["sZMode"]], dictScenario,
            daReferenceOffset, dSweepMedian)
        np.savetxt(f"fluxSamples_{sKey}.txt", daFlux)
        dictScenarios[sKey] = dictScenarioSummary
    np.savetxt("fluxSamplesForwardModelSweep.txt", daSweepFlux)

    dictBudget = {
        "dictLevel1ForwardModel": fdictDecomposeForwardModel(
            os.path.dirname(args.converged_flux)),
        "dictLevel2SourceIndices": fdictDecomposeSources(
            dictSourcesByMode["population"]),
        "dictScenarios": dictScenarios,
        "dictForwardModelSweep": fdictSummarizeFlux(
            daSweepFlux, "vconverge sweep (population z)"),
    }
    dictBudget["dictValidation"] = fdictValidate(dictBudget)
    dictBudget["dictGains"] = fdictComputeGains(dictScenarios)
    with open("uncertaintyBudget.json", "w") as fileHandle:
        json.dump(dictBudget, fileHandle, indent=2)
    fnPrintSummary(dictBudget)


def fdictComputeGains(dictScenarios):
    """Express each state of knowledge as a factor gained over no measurement."""
    dBaseline = dictScenarios["noMeasurement"]["sigma_log10"]
    return {sKey: {"dSpreadDex": dictScenario["sigma_log10"],
                   "dFactorVersusNoMeasurement":
                       dBaseline / dictScenario["sigma_log10"]}
            for sKey, dictScenario in dictScenarios.items()}


def fdictValidate(dictBudget):
    """Check the analytic propagation against the forward-model sweep."""
    dAnalytic = dictBudget["dictScenarios"]["noMeasurement"]["sigma_log10"]
    dSweep = dictBudget["dictForwardModelSweep"]["sigma_log10"]
    return {
        "sigma_log10_analytic": dAnalytic,
        "sigma_log10_forward_model_sweep": dSweep,
        "agreement_ratio": dAnalytic / dSweep,
        "note": "the analytic propagation must reproduce the forward-model "
                "sweep in the population case before its informed-case "
                "prediction can be trusted",
    }


def fnPrintSummary(dictBudget):
    """Print both levels of the budget and the value of the measurement."""
    dictL1 = dictBudget["dictLevel1ForwardModel"]
    print(f"LEVEL 1 - forward model ({dictL1['iTrials']} trials, "
          f"R^2 = {dictL1['dTotalRSquared']:.4f}):")
    for sBlock, dShare in sorted(dictL1["dictBlockShares"].items(),
                                 key=lambda t: -t[1]):
        print(f"  {sBlock:38s} {100 * dShare:6.2f}%")
    print("\nLEVEL 2 - within the X-UV coefficients "
          "(first-order indices, no measurement):")
    for sName, dictShare in sorted(
            dictBudget["dictLevel2SourceIndices"]["dictSourceShares"].items(),
            key=lambda t: -t[1]["dVarianceShare"]):
        print(f"  {sName:46s} {100 * dictShare['dVarianceShare']:6.2f}%  "
              f"({dictShare['dSpreadDex']:.3f} dex)")
    dictV = dictBudget["dictValidation"]
    print(f"\nValidation: analytic {dictV['sigma_log10_analytic']:.3f} dex vs "
          f"sweep {dictV['sigma_log10_forward_model_sweep']:.3f} dex "
          f"(ratio {dictV['agreement_ratio']:.2f})")
    print("\nSTATES OF KNOWLEDGE (projected cases assume the measured offsets "
          "persist):")
    for sKey, dictScenario in dictBudget["dictScenarios"].items():
        sMark = "*" if dictScenario["bProjected"] else " "
        dFactor = dictBudget["dictGains"][sKey]["dFactorVersusNoMeasurement"]
        print(f" {sMark}{dictScenario['label']:36s} sigma "
              f"{dictScenario['sigma_log10']:.3f} dex, median "
              f"{dictScenario['median']:.0f}, 95% CI "
              f"[{dictScenario['ci95'][0]:.0f}, "
              f"{dictScenario['ci95'][1]:.0f}]  ({dFactor:.2f}x)")


if __name__ == "__main__":
    main()
