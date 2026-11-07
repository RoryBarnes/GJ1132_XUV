#!/usr/bin/env python3
"""
Plot the MUSCLES X-ray to XUV band conversion for two stellar samples.

Each marker is one star's log10(L_XUV/L_bol) against log10(L_X/L_bol), both
integrated from its MUSCLES or Mega-MUSCLES SED. The solid lines are unweighted
fits to all stars and to the M dwarfs alone. Each band is the 1-sigma
uncertainty on the mean relation, propagated from the slope-intercept
covariance. The dashed line is the relation published by Engle (2024, ApJ 960,
62). Unweighted least squares assumes every star shares one unknown
uncertainty in log10(L_XUV/L_bol), estimated from the residuals as
sqrt(RSS/(N-2)) (Hogg, Bovy & Lang 2010, arXiv:1008.4686, Section 7); the
uniform vertical error bars show that estimate for the all-star fit. It is a
total per-star scatter, not a measurement error, and the fit treats
log10(L_X/L_bol) as exact.

Usage: python plotMusclesConversion.py <outputPath>
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
S_FIT_FILE = "conversionFit.json"
S_STAR_FILE = "perStarBandFluxes.json"
S_TARGET_STAR = "GJ 1132"
D_ABSCISSA_PADDING = 0.15
I_GRID_POINTS = 200

S_COLOR_ENGLE = "black"
S_COLOR_TARGET = vplot.colors.red
LIST_FIT_STYLES = [
    {"sKey": "primary_fit_all_targets_rosat_band", "sName": "All stars",
     "sColor": vplot.colors.dark_blue},
    {"sKey": "alternate_fit_m_dwarfs_only", "sName": "M dwarfs only",
     "sColor": vplot.colors.purple},
]
DICT_CLASS_STYLES = {
    "M": {"sMarker": "o", "sColor": vplot.colors.pale_blue, "sLabel": "M dwarfs"},
    "K": {"sMarker": "s", "sColor": vplot.colors.orange, "sLabel": "K dwarfs"},
}


def fdictLoadJson(sFileName):
    """Load a JSON product from the step directory, exiting cleanly on failure."""
    sPath = os.path.join(S_DIRECTORY, sFileName)
    try:
        with open(sPath) as fileHandle:
            return json.load(fileHandle)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"ERROR: could not read {sPath}: {error}. "
                 "Run dataDeriveMusclesConversion.py first.")


def fdaComputeMeanRelationVariance(daAbscissa, dictFit):
    """Propagate the slope-intercept covariance to the mean relation."""
    daCovariance = np.array(dictFit["covariance_slope_intercept"])
    return (daAbscissa**2 * daCovariance[0, 0]
            + 2.0 * daAbscissa * daCovariance[0, 1]
            + daCovariance[1, 1])


def fsFormatFitLabel(sName, dSlope, dIntercept):
    """Format a legend entry that states a relation's coefficients."""
    return f"{sName}: slope {dSlope:.3f}, intercept {dIntercept:.3f}"


def fdEstimateResidualSigma(dictFit):
    """Return the shared per-star uncertainty sqrt(RSS/(N-2)) of a fit."""
    iCount = dictFit["n_stars"]
    dSigmaMaximumLikelihood = dictFit["intrinsic_scatter_dex"]["fScatterMaximumLikelihood"]
    return dSigmaMaximumLikelihood * np.sqrt(iCount / (iCount - 2.0))


def fnPlotFit(axis, daAbscissa, dictFit, dictStyle):
    """Draw one fitted line with its 1-sigma mean-relation band."""
    daLine = dictFit["slope"] * daAbscissa + dictFit["intercept"]
    daMeanWidth = np.sqrt(fdaComputeMeanRelationVariance(daAbscissa, dictFit))
    sColor = dictStyle["sColor"]
    axis.fill_between(daAbscissa, daLine - daMeanWidth, daLine + daMeanWidth,
                      color=sColor, alpha=0.22, linewidth=0,
                      label=r"1$\sigma$ uncertainty on mean relation")
    sName = f"{dictStyle['sName']} (N = {dictFit['n_stars']})"
    axis.plot(daAbscissa, daLine, color=sColor, linewidth=2.0,
              label=fsFormatFitLabel(sName, dictFit["slope"], dictFit["intercept"]))


def fnPlotEngleRelation(axis, daAbscissa, dictEngle):
    """Draw the relation published by Engle (2024)."""
    dSlope = dictEngle["engle_slope"]
    dIntercept = dictEngle["engle_intercept"]
    axis.plot(daAbscissa, dSlope * daAbscissa + dIntercept,
              color=S_COLOR_ENGLE, linestyle="--", linewidth=1.6,
              label=fsFormatFitLabel("Engle (2024)", dSlope, dIntercept))


def fnPlotUniformErrorBars(axis, listStars, tSigmas):
    """Draw one shared vertical error bar on every star."""
    axis.errorbar([dictStar["log_lx_lbol"] for dictStar in listStars],
                  [dictStar["log_lxuv_lbol"] for dictStar in listStars],
                  yerr=tSigmas[0], fmt="none", ecolor="0.55", elinewidth=1.0,
                  zorder=3, label=(fr"uniform $\pm${tSigmas[0]:.3f} dex "
                                   fr"(M-dwarf fit: $\pm${tSigmas[1]:.3f})"))


def fnPlotStars(axis, listStars):
    """Scatter every comparison star as a filled marker, grouped by class."""
    for sClass, dictStyle in DICT_CLASS_STYLES.items():
        listSelected = [dictStar for dictStar in listStars
                        if dictStar["spectral_class_group"] == sClass
                        and dictStar["star"] != S_TARGET_STAR]
        axis.scatter([dictStar["log_lx_lbol"] for dictStar in listSelected],
                     [dictStar["log_lxuv_lbol"] for dictStar in listSelected],
                     s=52, marker=dictStyle["sMarker"], zorder=4,
                     color=dictStyle["sColor"], edgecolors="white",
                     linewidths=0.6, label=dictStyle["sLabel"])


def fnPlotTargetStar(axis, listStars):
    """Mark and directly label the planet's host star."""
    dictTarget = next(dictStar for dictStar in listStars
                      if dictStar["star"] == S_TARGET_STAR)
    tPoint = (dictTarget["log_lx_lbol"], dictTarget["log_lxuv_lbol"])
    axis.scatter(*tPoint, s=190, marker="*", color=S_COLOR_TARGET,
                 edgecolors="white", linewidths=0.8, zorder=5)
    axis.annotate(S_TARGET_STAR, tPoint, xytext=(-8, 9),
                  textcoords="offset points", ha="right", fontsize=11)


def fnFinishAxis(axis, daAbscissa):
    """Set the limits, axis labels and legend."""
    axis.set_xlim(daAbscissa[0], daAbscissa[-1])
    axis.set_xlabel(r"$\log_{10}(L_{\rm X}/L_{\rm bol})$  [5.17$-$124 $\rm\AA$]")
    axis.set_ylabel(r"$\log_{10}(L_{\rm XUV}/L_{\rm bol})$  [5$-$1700 $\rm\AA$]")
    axis.legend(loc="upper left", fontsize=8)


def fnRenderFigure(sOutputPath, dictReport, listStars):
    """Compose the conversion figure and save it to disk."""
    daStarAbscissa = [dictStar["log_lx_lbol"] for dictStar in listStars]
    daAbscissa = np.linspace(min(daStarAbscissa) - D_ABSCISSA_PADDING,
                             max(daStarAbscissa) + D_ABSCISSA_PADDING,
                             I_GRID_POINTS)
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for dictStyle in LIST_FIT_STYLES:
        fnPlotFit(axis, daAbscissa, dictReport[dictStyle["sKey"]], dictStyle)
    fnPlotEngleRelation(axis, daAbscissa, dictReport["engle_comparison"])
    tSigmas = tuple(fdEstimateResidualSigma(dictReport[dictStyle["sKey"]])
                    for dictStyle in LIST_FIT_STYLES)
    fnPlotUniformErrorBars(axis, listStars, tSigmas)
    fnPlotStars(axis, listStars)
    fnPlotTargetStar(axis, listStars)
    fnFinishAxis(axis, daAbscissa)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")


def main():
    """Render the MUSCLES conversion figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sOutputPath", help="Path of the figure to write.")
    sOutputPath = parser.parse_args().sOutputPath
    dictReport = fdictLoadJson(S_FIT_FILE)
    listStars = fdictLoadJson(S_STAR_FILE)["stars"]
    os.makedirs(os.path.dirname(os.path.abspath(sOutputPath)), exist_ok=True)
    fnRenderFigure(sOutputPath, dictReport, listStars)
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
