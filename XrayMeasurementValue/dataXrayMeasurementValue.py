#!/usr/bin/env python3
"""
Quantify what one X-ray measurement of the host star is worth.

Decomposes the cumulative-XUV uncertainty budget for GJ 1132 b into its four
independent components and propagates it under two states of knowledge that
differ ONLY in the star's offset z within the activity relation's population
scatter:

  population z ~ N(0, 1)          -- no host-star X-ray measurement exists
  informed   z ~ p(z | L_X, age)  -- the measured L_X locates the star

Both use the identical rotation-only age prior, so the comparison isolates the
value of the measurement itself. The four components are

  A  refit mean line (Engle coefficients with their full covariance)
  B  MUSCLES band-conversion coefficients (slope/intercept covariance)
  C  MUSCLES band-conversion intrinsic scatter (SED-to-SED dispersion)
  D  population intrinsic scatter, z * sigma_int(tau), converted to the X-UV band

Because the flux scales as 10^offset, the propagated distribution is lognormal:
its MEDIAN tracks the central relation while its MEAN is inflated by
exp(sigma_ln^2 / 2). Reporting the mean alone conflates a wider error bar with
a larger flux; this step separates them.

The analytic propagation is VALIDATED against the full vconverge forward-model
sweep in the population case (the step's converged-flux input); the validated
model then predicts the informed case, which the Z-Offset Cumulative XUV
Comparison step confirms with an independent forward-model run.

Reference: Kelly (2007) ApJ 665, 1489 for the measurement-error framework.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (fdictLoadConvergedJson, daExtractFluxValues,
                                 D_SHORELINE_FLUX)

I_SEED = 42
I_NUM_SAMPLES = 200000
D_TAU_GJ1132 = 0.8533


def fdaSampleMeanLine(daRows, dTau):
    """Component A: the refit hinge evaluated at dTau, in the native X-ray band."""
    return (daRows[:, 6] * dTau + daRows[:, 7]
            + daRows[:, 8] * np.clip(dTau - daRows[:, 9], 0.0, None))


def fdaSampleScatterLaw(daRows, dTau, dictSummary):
    """Return each posterior row's native-band sigma_int evaluated at dTau."""
    return np.exp(daRows[:, 10] + daRows[:, 11]
                  * (dTau - dictSummary["dPivotTau"])
                  / dictSummary["dScaleTau"])


def ftSampleConversion(dictConversion, iCount):
    """Components B and C: draw conversion coefficients and their scatter."""
    daDraws = np.random.multivariate_normal(
        [dictConversion["slope"], dictConversion["intercept"]],
        np.array(dictConversion["covariance_slope_intercept"]), iCount)
    dScatter = dictConversion["intrinsic_scatter_dex"]["fScatterPosteriorMedian"]
    return daDraws[:, 0], daDraws[:, 1], np.random.normal(0, dScatter, iCount)


def fdaDrawOffsetZ(sMode, daInformedPosterior, iCount):
    """Component D's z: the population prior or the L_X-informed posterior."""
    if sMode == "population":
        return np.random.normal(0.0, 1.0, iCount)
    return daInformedPosterior[np.random.randint(0, len(daInformedPosterior),
                                                 iCount)]


def fdaPredictXuvOffset(daChain, dictSummary, dictConversion,
                        daInformedPosterior, sMode):
    """Return X-UV-band log-flux offsets under one state of knowledge."""
    daRows = daChain[np.random.randint(0, len(daChain), I_NUM_SAMPLES)]
    daNative = fdaSampleMeanLine(daRows, D_TAU_GJ1132)
    daSigmaInt = fdaSampleScatterLaw(daRows, D_TAU_GJ1132, dictSummary)
    daZ = fdaDrawOffsetZ(sMode, daInformedPosterior, I_NUM_SAMPLES)
    daSlope, daIntercept, daScatter = ftSampleConversion(dictConversion,
                                                         I_NUM_SAMPLES)
    return daSlope * (daNative + daZ * daSigmaInt) + daIntercept + daScatter


def fdictDecomposeBudget(daChain, dictSummary, dictConversion,
                         daInformedPosterior):
    """Return the per-component spreads (dex) of the X-UV-band offset."""
    daRows = daChain[np.random.randint(0, len(daChain), I_NUM_SAMPLES)]
    daNative = fdaSampleMeanLine(daRows, D_TAU_GJ1132)
    daSigmaInt = fdaSampleScatterLaw(daRows, D_TAU_GJ1132, dictSummary)
    dSlope = dictConversion["slope"]
    daCovariance = np.array(dictConversion["covariance_slope_intercept"])
    dNative = float(np.mean(daNative))
    return {
        "A_refit_mean_line": float(np.std(daNative) * dSlope),
        "B_conversion_covariance": float(np.sqrt(
            dNative ** 2 * daCovariance[0, 0] + daCovariance[1, 1]
            + 2 * dNative * daCovariance[0, 1])),
        "C_conversion_scatter": float(
            dictConversion["intrinsic_scatter_dex"]["fScatterPosteriorMedian"]),
        "D_population_scatter_population_z": float(
            np.mean(daSigmaInt) * dSlope),
        "D_population_scatter_informed_z": float(
            np.mean(daSigmaInt) * dSlope * np.std(daInformedPosterior)),
        "sigma_int_native_at_gj1132_age": float(np.mean(daSigmaInt)),
    }


def fdictSummarizeFlux(daFlux, sLabel):
    """Return lognormal-aware summary statistics for a flux distribution."""
    daLog = np.log10(daFlux)
    return {
        "label": sLabel,
        "median": float(np.median(daFlux)),
        "mean": float(np.mean(daFlux)),
        "ci95": [float(np.percentile(daFlux, 2.5)),
                 float(np.percentile(daFlux, 97.5))],
        "sigma_log10": float(np.std(daLog)),
        "mean_over_median": float(np.mean(daFlux) / np.median(daFlux)),
        "fraction_above_shoreline": float(np.mean(daFlux > D_SHORELINE_FLUX)),
    }


def fdictValidateAgainstSweep(daSweepFlux, daPopulationOffset):
    """Compare the analytic population spread with the forward-model sweep."""
    dAnalytic = float(np.std(daPopulationOffset))
    dSweep = float(np.std(np.log10(daSweepFlux)))
    return {
        "sigma_log10_analytic": dAnalytic,
        "sigma_log10_forward_model_sweep": dSweep,
        "agreement_ratio": dAnalytic / dSweep,
        "note": "The analytic propagation is validated against the vconverge "
                "sweep in the population case before predicting the informed "
                "case.",
    }


def fdaAnchorFlux(daOffset, daPopulationOffset, dSweepMedian):
    """Convert log-offsets to flux, anchored on the sweep's population median."""
    return dSweepMedian * 10.0 ** (daOffset - np.median(daPopulationOffset))


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Value of one host-star X-ray measurement for the "
                    "cumulative XUV flux on GJ 1132 b.")
    parser.add_argument("--joint-chain", required=True)
    parser.add_argument("--fit-summary", required=True)
    parser.add_argument("--conversion-fit", required=True)
    parser.add_argument("--z-offsets", required=True,
                        help="L_X-informed z posterior samples (.txt).")
    parser.add_argument("--converged-flux", required=True,
                        help="Population-z vconverge output to validate and "
                             "anchor against (Converged_Param_Dictionary.json).")
    return parser.parse_args()


def main():
    """Compute the uncertainty budget and both flux distributions."""
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

    daPopulationOffset = fdaPredictXuvOffset(
        daChain, dictSummary, dictConversion, daInformed, "population")
    daInformedOffset = fdaPredictXuvOffset(
        daChain, dictSummary, dictConversion, daInformed, "informed")
    dSweepMedian = float(np.median(daSweepFlux))
    daFluxPopulation = fdaAnchorFlux(daPopulationOffset, daPopulationOffset,
                                     dSweepMedian)
    daFluxInformed = fdaAnchorFlux(daInformedOffset, daPopulationOffset,
                                   dSweepMedian)

    np.savetxt("fluxSamplesPopulationZ.txt", daFluxPopulation)
    np.savetxt("fluxSamplesInformedZ.txt", daFluxInformed)
    np.savetxt("fluxSamplesForwardModelSweep.txt", daSweepFlux)
    dictBudget = {
        "dictComponentsDex": fdictDecomposeBudget(
            daChain, dictSummary, dictConversion, daInformed),
        "dictValidation": fdictValidateAgainstSweep(daSweepFlux,
                                                    daPopulationOffset),
        "dictPopulationZ": fdictSummarizeFlux(daFluxPopulation,
                                              "no X-ray measurement"),
        "dictInformedZ": fdictSummarizeFlux(daFluxInformed,
                                            "with X-ray measurement"),
        "dictForwardModelSweep": fdictSummarizeFlux(daSweepFlux,
                                                    "vconverge sweep "
                                                    "(population z)"),
    }
    dictBudget["dSpreadReductionFactor"] = (
        dictBudget["dictPopulationZ"]["sigma_log10"]
        / dictBudget["dictInformedZ"]["sigma_log10"])
    with open("uncertaintyBudget.json", "w") as fileHandle:
        json.dump(dictBudget, fileHandle, indent=2)
    fnPrintSummary(dictBudget)


def fnPrintSummary(dictBudget):
    """Print the uncertainty budget and the value of the measurement."""
    dictC = dictBudget["dictComponentsDex"]
    print("Uncertainty budget at GJ 1132's age [dex, X-UV band]:")
    for sKey in ("A_refit_mean_line", "B_conversion_covariance",
                 "C_conversion_scatter", "D_population_scatter_population_z"):
        print(f"  {sKey:34s} {dictC[sKey]:.3f}")
    dictV = dictBudget["dictValidation"]
    print(f"Validation: analytic {dictV['sigma_log10_analytic']:.3f} dex vs "
          f"sweep {dictV['sigma_log10_forward_model_sweep']:.3f} dex "
          f"(ratio {dictV['agreement_ratio']:.2f})")
    for sKey in ("dictPopulationZ", "dictInformedZ"):
        dictS = dictBudget[sKey]
        print(f"{dictS['label']:26s}: median {dictS['median']:.0f}, "
              f"95% CI [{dictS['ci95'][0]:.0f}, {dictS['ci95'][1]:.0f}], "
              f"sigma {dictS['sigma_log10']:.3f} dex, "
              f"P(>shoreline) {dictS['fraction_above_shoreline']:.3f}")
    print(f"One X-ray measurement narrows the flux by a factor "
          f"{dictBudget['dSpreadReductionFactor']:.2f} in log-spread.")


if __name__ == "__main__":
    main()
