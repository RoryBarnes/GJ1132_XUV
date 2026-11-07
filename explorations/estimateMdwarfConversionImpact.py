#!/usr/bin/env python3
"""
Estimate how an M-dwarf-only band conversion would change GJ 1132 b's XUV dose.

This is a screening estimate, not a replacement for rerunning the vconverge
sweeps. It evaluates the Engle hinge that vplanet's fdLXUVEngle uses,

    log10(L_XUV/L_bol) = s (a tau + b + c max(tau - d, 0) + z sigma) + i + n,

with tau = log10(age/Gyr), and integrates L_XUV(t) over one fixed L_bol(t)
history from the forward model. Stellar mass, present age, and planet inputs
are held fixed because the existing sweep attributes only 0.5% of the flux
variance to them. Both conversions share the same random deviates (common
random numbers; Glasserman 2003, Monte Carlo Methods in Financial Engineering,
Section 4.2), so their ratio is free of independent sampling noise. Flux is
anchored to the median of the existing no-measurement sweep.

Usage: python estimateMdwarfConversionImpact.py --joint-chain <npy>
       --fit-summary <json> --conversion-fit <json> --z-offsets <txt>
       --star-forward <file> --budget <json> --output <json>
"""

import argparse
import json

import numpy as np

I_SEED = 42
I_DRAWS = 50000
I_AGE_GRID_POINTS = 400
D_TAU_PRESENT = 0.8533
D_TAU_REFERENCE = 0.85
D_INITIAL_AGE_YEARS = 1.0e6
D_SHORELINE_FLUX = 51.43
DICT_CONVERSIONS = {"allStars": "alternate_fit_all_targets_rosat_band",
                    "mDwarfsOnly": "adopted_fit"}


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for sName in ("joint-chain", "fit-summary", "conversion-fit", "z-offsets",
                  "star-forward", "budget", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdictLoadJson(sPath):
    """Load one JSON file."""
    with open(sPath) as fileHandle:
        return json.load(fileHandle)


def ftBuildAgeGrid(sStarForwardPath):
    """Return log-spaced tau, age widths, and L_bol from the forward history."""
    daForward = np.loadtxt(sStarForwardPath, usecols=(0, 1))
    daForwardAge = daForward[:, 0] + D_INITIAL_AGE_YEARS
    daAge = np.logspace(np.log10(daForwardAge[0]), D_TAU_PRESENT + 9.0,
                        I_AGE_GRID_POINTS)
    daLuminosity = np.interp(daAge, daForwardAge, daForward[:, 1])
    return np.log10(daAge / 1.0e9), np.gradient(daAge), daLuminosity


def fdictDrawDeviates(iRows, daInformedZ):
    """Draw the deviates that both conversions share."""
    return {
        "iaRows": np.random.randint(0, iRows, I_DRAWS),
        "daPopulationZ": np.random.normal(0.0, 1.0, I_DRAWS),
        "daInformedZ": daInformedZ[np.random.randint(0, len(daInformedZ), I_DRAWS)],
        "daaConversion": np.random.normal(0.0, 1.0, (I_DRAWS, 2)),
        "daScatter": np.random.normal(0.0, 1.0, I_DRAWS),
    }


def ftDrawConversion(dictFit, dictDeviates):
    """Return slope, intercept, and scatter draws for one conversion fit."""
    daFactor = np.linalg.cholesky(np.array(dictFit["covariance_slope_intercept"]))
    daaDraws = (np.array([dictFit["slope"], dictFit["intercept"]])
                + dictDeviates["daaConversion"] @ daFactor.T)
    dScatter = dictFit["intrinsic_scatter_dex"]["fScatterPosteriorMedian"]
    return daaDraws[:, 0], daaDraws[:, 1], dScatter * dictDeviates["daScatter"]


def fdaIntegrateDose(daaRows, daOffset, dictConversionDraws, tGrid):
    """Integrate L_bol(t) 10^(hinge) over age for every draw, up to a constant."""
    daTau, daWidth, daLuminosity = tGrid
    daSlope, daIntercept, daScatter = dictConversionDraws
    daDose = np.empty(I_DRAWS)
    for iDraw in range(I_DRAWS):
        daNative = (daaRows[iDraw, 6] * daTau + daaRows[iDraw, 7] + daOffset[iDraw]
                    + daaRows[iDraw, 8] * np.clip(daTau - daaRows[iDraw, 9], 0, None))
        daFraction = 10.0 ** (daSlope[iDraw] * daNative + daIntercept[iDraw]
                              + daScatter[iDraw])
        daDose[iDraw] = np.sum(daLuminosity * daFraction * daWidth)
    return daDose


def fdictSummarizeFlux(daFlux):
    """Summarize one lognormal-like flux distribution."""
    return {"dMedian": float(np.median(daFlux)), "dMean": float(np.mean(daFlux)),
            "daCi95": [float(np.percentile(daFlux, 2.5)),
                       float(np.percentile(daFlux, 97.5))],
            "dSigmaLog10": float(np.std(np.log10(daFlux))),
            "dFractionAboveShoreline": float(np.mean(daFlux > D_SHORELINE_FLUX))}


def fdictCompareScenario(dictBase, dictAlternate):
    """Report the fractional change of the alternate conversion's statistics."""
    return {"dMedianRatio": dictAlternate["dMedian"] / dictBase["dMedian"],
            "dMeanRatio": dictAlternate["dMean"] / dictBase["dMean"],
            "dMedianShiftDex": float(np.log10(dictAlternate["dMedian"]
                                              / dictBase["dMedian"]))}


def fdaScatterOffset(daaRows, daZ, dictSummary):
    """Return each draw's native-band offset z sigma_int at the reference age."""
    return daZ * np.exp(daaRows[:, 10] + daaRows[:, 11]
                        * (D_TAU_REFERENCE - dictSummary["dPivotTau"])
                        / dictSummary["dScaleTau"])


def fdictEvaluateAll(args):
    """Evaluate both conversions in both states of knowledge."""
    daChain = np.load(args.joint_chain)
    dictSummary = fdictLoadJson(args.fit_summary)
    dictReport = fdictLoadJson(args.conversion_fit)
    dictDeviates = fdictDrawDeviates(len(daChain), np.loadtxt(args.z_offsets))
    daaRows = daChain[dictDeviates["iaRows"]]
    tGrid = ftBuildAgeGrid(args.star_forward)
    dictDoses = {}
    for sScenario, sZKey in (("noMeasurement", "daPopulationZ"),
                             ("xrayMeasurement", "daInformedZ")):
        daOffset = fdaScatterOffset(daaRows, dictDeviates[sZKey], dictSummary)
        for sConversion, sFitKey in DICT_CONVERSIONS.items():
            dictDraws = ftDrawConversion(dictReport[sFitKey], dictDeviates)
            dictDoses[(sScenario, sConversion)] = fdaIntegrateDose(
                daaRows, daOffset, dictDraws, tGrid)
    return dictDoses


def fdictBuildReport(dictDoses, dSweepMedian):
    """Anchor every dose to the sweep median and tabulate the comparison."""
    dScale = dSweepMedian / np.median(dictDoses[("noMeasurement", "allStars")])
    dictReport = {"sNote": "Screening estimate; see the module docstring."}
    for sScenario in ("noMeasurement", "xrayMeasurement"):
        dictBase = fdictSummarizeFlux(dScale * dictDoses[(sScenario, "allStars")])
        dictAlternate = fdictSummarizeFlux(dScale * dictDoses[(sScenario, "mDwarfsOnly")])
        dictReport[sScenario] = {
            "allStars": dictBase, "mDwarfsOnly": dictAlternate,
            "comparison": fdictCompareScenario(dictBase, dictAlternate)}
    return dictReport


def main():
    """Write the conversion-impact estimate to a JSON file."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    dSweepMedian = fdictLoadJson(args.budget)["dictScenarios"]["noMeasurement"]["median"]
    dictReport = fdictBuildReport(fdictEvaluateAll(args), dSweepMedian)
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
