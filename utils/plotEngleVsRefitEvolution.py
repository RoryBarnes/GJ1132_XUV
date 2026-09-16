#!/usr/bin/env python3
"""
Compare the published Engle relations with the joint hierarchical refit.

Two panels per fit variant. Left: the rotation-age relation with the benchmark
data, the published segmented fit, and the refit posterior (median, 68%
mean-line band, and the inferred intrinsic-scatter envelope). Right: the
X-UV(5-1700 A) activity-age relation - the refit is performed in native
log(L_X/L_bol) and converted to the X-UV band through the re-derived MUSCLES
conversion (with its covariance sampled), so the comparison against Engle's
published X-UV coefficients is band-consistent. Field stars are plotted at
their posterior-median rotation ages (display only; the fit never uses
tabulated ages for them).

Usage: python plotEngleVsRefitEvolution.py <tag> [outputPath]
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot  # noqa: F401  (applies the project figure style on import)

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
I_CURVE_DRAWS = 2000

DICT_PUBLISHED_XUV = {
    "midlate": [-0.1456, -2.8876, -1.8187, 0.3545],
    "early": [-0.4896, -3.2128, -0.4469, -0.2985],
}
DICT_PUBLISHED_ROTATION = {
    "midlate": [0.0561, -0.8900, -0.0521, 24.1888],
    "early": [0.0621, -1.0437, -0.0528, 23.4933],
}


def fdaHinge(daCoefficients, daX):
    """Evaluate the two-segment hinge for one coefficient vector."""
    dA, dB, dC, dD = daCoefficients
    return dA * daX + dB + dC * np.where(daX >= dD, daX - dD, 0.0)


def fdaHingeRows(daRows, daX):
    """Evaluate the hinge for many posterior rows on a common grid."""
    daGrid = daRows[:, 0:1] * daX + daRows[:, 1:2]
    return daGrid + daRows[:, 2:3] * np.clip(daX - daRows[:, 3:4], 0.0, None)


def ftLoadInputs(sTag):
    """Load the chain, fit summary, sample records, and MUSCLES conversion."""
    daChain = np.load(os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy"))
    with open(os.path.join(S_DIRECTORY,
                           f"jointFitSummary_{sTag}.json")) as fileHandle:
        dictSummary = json.load(fileHandle)
    sSampleFile = ("jointSampleMidLate.json" if sTag.startswith("midlate")
                   else "jointSampleEarly.json")
    with open(os.path.join(S_DIRECTORY, sSampleFile)) as fileHandle:
        listRecords = json.load(fileHandle)
    with open(os.path.join(S_DIRECTORY, "musclesConversion",
                           "conversionFit.json")) as fileHandle:
        dictPrimary = json.load(fileHandle)["primary_fit_all_targets_rosat_band"]
    dictConversion = {"dSlope": dictPrimary["slope"],
                      "dIntercept": dictPrimary["intercept"],
                      "daCovariance": dictPrimary["covariance_slope_intercept"]}
    return daChain, dictSummary, listRecords, dictConversion


def fdaScatterRows(daChain, iOffset, daX, dPivot, dScale):
    """Evaluate each posterior row's log-linear scatter law on a grid."""
    daLogSigma = (daChain[:, iOffset:iOffset + 1]
                  + daChain[:, iOffset + 1:iOffset + 2] * (daX - dPivot) / dScale)
    return np.exp(daLogSigma)


def fnPlotBand(axis, daX, daCurves, sColor, sLabel):
    """Plot the median curve and 68% band of a family of posterior curves."""
    daMedian = np.percentile(daCurves, 50, axis=0)
    axis.fill_between(daX, np.percentile(daCurves, 16, axis=0),
                      np.percentile(daCurves, 84, axis=0),
                      color=sColor, alpha=0.3, linewidth=0)
    axis.plot(daX, daMedian, color=sColor, linewidth=2, label=sLabel)
    return daMedian


def fnPlotRotationPanel(axis, daChain, dictSummary, listRecords, sTag):
    """Render the rotation-age panel: data, published fit, refit bands."""
    daRows = daChain[np.random.randint(0, len(daChain), I_CURVE_DRAWS)]
    daProtGrid = np.linspace(0.3, 180.0, 400)
    daCurves = fdaHingeRows(daRows[:, 0:4], daProtGrid)
    daMedian = fnPlotBand(axis, daProtGrid, daCurves, "C0", "joint refit")
    daSigma = np.median(fdaScatterRows(daRows, 4, daProtGrid,
                                       dictSummary["dPivotProt"],
                                       dictSummary["dScaleProt"]), axis=0)
    axis.fill_between(daProtGrid, daMedian - daSigma, daMedian + daSigma,
                      color="C0", alpha=0.12, linewidth=0)
    axis.plot(daProtGrid, fdaHinge(
        DICT_PUBLISHED_ROTATION[sTag.split("_")[0]], daProtGrid),
        color="C1", linewidth=2, linestyle="--", label="Engle published")
    fnPlotBenchmarkData(axis, listRecords)
    axis.set_xlabel("rotation period [d]")
    axis.set_ylabel(r"$\log_{10}$ age [Gyr]")
    axis.set_ylim(-1.4, 1.4)
    axis.legend(loc="lower right", fontsize=9)


def fnPlotBenchmarkData(axis, listRecords):
    """Overlay the rotation benchmarks, marking subdwarfs distinctly."""
    for dictRecord in listRecords:
        if dictRecord["sAgeProvenance"] != "independent" \
                or dictRecord["dictRotation"] is None \
                or dictRecord["bExcludedFromPaperFits"]:
            continue
        dictAge = dictRecord["dictAgeConstraint"]
        dTau = np.log10(dictAge["dValue"])
        sMarker = "s" if dictRecord["bSubdwarf"] else "o"
        axis.errorbar(dictRecord["dictRotation"]["dProtDays"], dTau,
                      yerr=[[dTau - np.log10(max(
                          dictAge["dValue"] - dictAge["dMinus"], 1e-3))],
                          [np.log10(dictAge["dValue"] + dictAge["dPlus"])
                           - dTau]],
                      fmt=sMarker, color="0.35", markersize=4,
                      elinewidth=0.8, capsize=0)


def fdaConvertRowsToXuv(daActivityCurves, dictConversion, iDraws):
    """Convert native log(Lx/Lbol) curves to the X-UV band with covariance."""
    daMeanCoefficients = np.array([dictConversion["dSlope"],
                                   dictConversion["dIntercept"]])
    daCovariance = np.array(dictConversion["daCovariance"])
    daDraws = np.random.multivariate_normal(daMeanCoefficients, daCovariance,
                                            iDraws)
    return (daDraws[:, 0:1] * daActivityCurves + daDraws[:, 1:2])


def fnPlotXuvPanel(axis, daChain, dictSummary, listRecords, dictConversion,
                   sTag):
    """Render the X-UV activity-age panel, band-consistently."""
    daRows = daChain[np.random.randint(0, len(daChain), I_CURVE_DRAWS)]
    daTauGrid = np.linspace(-1.1, 1.2, 400)
    daNative = fdaHingeRows(daRows[:, 6:10], daTauGrid)
    daXuvCurves = fdaConvertRowsToXuv(daNative, dictConversion, I_CURVE_DRAWS)
    daMedian = fnPlotBand(axis, daTauGrid, daXuvCurves, "C0",
                          "joint refit + MUSCLES conversion")
    daSigma = np.median(fdaScatterRows(daRows, 10, daTauGrid,
                                       dictSummary["dPivotTau"],
                                       dictSummary["dScaleTau"]), axis=0)
    dSlope = dictConversion["dSlope"]
    axis.fill_between(daTauGrid, daMedian - dSlope * daSigma,
                      daMedian + dSlope * daSigma,
                      color="C0", alpha=0.12, linewidth=0)
    axis.plot(daTauGrid, fdaHinge(DICT_PUBLISHED_XUV[sTag.split("_")[0]],
                                  daTauGrid),
              color="C1", linewidth=2, linestyle="--", label="Engle published")
    fnPlotActivityData(axis, daChain, listRecords, dictConversion)
    axis.set_xlabel(r"$\log_{10}$ age [Gyr]")
    axis.set_ylabel(r"$\log_{10}(L_{\rm XUV}/L_{\rm bol})$ (5-1700 $\AA$)")
    axis.legend(loc="lower left", fontsize=9)


def fnPlotActivityData(axis, daChain, listRecords, dictConversion):
    """Overlay the X-ray stars, converted to the X-UV band, at display ages."""
    daMedianRotation = np.median(daChain[:, 0:4], axis=0)
    dSlope, dIntercept = dictConversion["dSlope"], dictConversion["dIntercept"]
    for dictRecord in listRecords:
        if dictRecord["dictXray"] is None:
            continue
        dTau = fdDisplayAge(dictRecord, daMedianRotation)
        if dTau is None:
            continue
        dYXuv = dSlope * dictRecord["dictXray"]["dLogLxLbol"] + dIntercept
        axis.errorbar(dTau, dYXuv,
                      yerr=dSlope * dictRecord["dictXray"]["dError"],
                      fmt="o", color="0.35", markersize=3, elinewidth=0.6,
                      capsize=0, alpha=0.7)


def fdDisplayAge(dictRecord, daMedianRotation):
    """Return the display log-age for one X-ray star (never used in the fit)."""
    if dictRecord["bClusterEnsemble"]:
        return dictRecord["dictAgeConstraint"]["dValue"]
    if dictRecord["dictRotation"] is None:
        return None
    if dictRecord["sAgeProvenance"] == "independent":
        return np.log10(dictRecord["dictAgeConstraint"]["dValue"])
    return float(fdaHinge(daMedianRotation,
                          np.array([dictRecord["dictRotation"]["dProtDays"]]))[0])


def main():
    """Render the two-panel Engle-vs-refit comparison for one fit variant."""
    np.random.seed(7)
    sTag = sys.argv[1]
    sOutputPath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        S_DIRECTORY, "Plot", f"EngleVsRefit_{sTag}.pdf")
    daChain, dictSummary, listRecords, dictConversion = ftLoadInputs(sTag)
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, (axisLeft, axisRight) = plt.subplots(1, 2, figsize=(11, 4.5))
    fnPlotRotationPanel(axisLeft, daChain, dictSummary, listRecords, sTag)
    fnPlotXuvPanel(axisRight, daChain, dictSummary, listRecords,
                   dictConversion, sTag)
    figure.suptitle(f"Engle relations: published vs joint refit ({sTag})",
                    fontsize=12)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
