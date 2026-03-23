#!/usr/bin/env python3
"""
Plot sampler comparison corner plot from pre-computed posterior samples.

Reads emcee, dynesty, multinest, and ultranest .npz sample files from the
output/ directory, overlays their contours on a corner plot, and adds
priors and maximum likelihood points.

Usage:
    python plotSamplerComparison.py output/sampler_comparison.pdf
"""

import os
import sys

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from pathlib import Path

import matplotlib.pyplot as plt
import vplot
import corner

# =========================================================================
# Configuration (must match dataBayesianPosteriors.py)
# =========================================================================

listParamLabels = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

listBounds = [
    (0.17, 0.22),
    (-4.0, -2.15),
    (0.1, 5.0),
    (1.0, 13.0),
    (0.4, 2.1),
]

listPriorData = [
    (0.1945, 0.0048, 0.0046),
    (-2.92, 0.26),
    (None, None),
    "empirical",
    (1.18, 0.31),
]

I_NUM_DIMENSIONS = len(listBounds)

D_TICK_FONTSIZE = 14
D_LABEL_FONTSIZE = 20
D_LEGEND_FONTSIZE = 20
D_FIGSIZE = 12


# =========================================================================
# Sample Loading
# =========================================================================


def fdaLoadSamplerSamples(sSamplesDir, sFilename):
    """Load posterior samples from a .npz file."""
    sPath = os.path.join(sSamplesDir, sFilename)
    if not os.path.exists(sPath):
        raise FileNotFoundError(f"Sample file not found: {sPath}")
    return np.load(sPath)["samples"]


def fkdeBuildAgePrior(sAgeSamplesPath, tBounds):
    """Load age samples and return a KDE for the empirical age prior."""
    daAgeGyr = np.loadtxt(sAgeSamplesPath) / 1e9
    daAgeGyr = daAgeGyr[(daAgeGyr >= tBounds[0]) & (daAgeGyr <= tBounds[1])]
    return gaussian_kde(daAgeGyr)


# =========================================================================
# Prior Evaluation
# =========================================================================


def fdaGaussian(daX, dMean, dStd):
    """Evaluate a normalized Gaussian PDF."""
    return (1.0 / (dStd * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((daX - dMean) / dStd) ** 2)


def fdaAsymmetricGaussian(daX, dMean, dStdPos, dStdNeg):
    """Evaluate a normalized asymmetric Gaussian PDF."""
    daResult = np.empty_like(daX)
    bUpper = daX >= dMean
    daResult[bUpper] = fdaGaussian(daX[bUpper], dMean, dStdPos)
    daResult[~bUpper] = fdaGaussian(daX[~bUpper], dMean, dStdNeg)
    return daResult


def fdaPriorDensity(iDimension, daXrange, kdeAgePrior):
    """Evaluate the prior density for a single dimension."""
    tPrior = listPriorData[iDimension]
    if tPrior == "empirical":
        return kdeAgePrior(daXrange)
    if tPrior[0] is None:
        dXmin, dXmax = listBounds[iDimension]
        return np.ones_like(daXrange) / (dXmax - dXmin)
    if len(tPrior) == 2:
        return fdaGaussian(daXrange, tPrior[0], tPrior[1])
    return fdaAsymmetricGaussian(daXrange, *tPrior)


# =========================================================================
# MaxLEV Results Reader
# =========================================================================


def fdaReadMaxLevParams(sMaxLevPath):
    """Read MAP parameters from a MaxLEV results file, or return None."""
    import re
    dictParamIndex = {
        "star.dMass": 0, "dMass": 0,
        "star.dSatXUVFrac": 1, "dSatXUVFrac": 1,
        "star.dSatXUVTime": 2, "dSatXUVTime": 2,
        "vpl.dStopTime": 3, "dStopTime": 3,
        "star.dXUVBeta": 4, "dXUVBeta": 4,
    }
    if not os.path.exists(sMaxLevPath):
        return None
    daParams = np.full(I_NUM_DIMENSIONS, np.nan)
    with open(sMaxLevPath, "r") as fileHandle:
        for sLine in fileHandle:
            matchParam = re.match(r"^(\S+)\s+=\s+(\S+)", sLine.strip())
            if matchParam and matchParam.group(1) in dictParamIndex:
                iIdx = dictParamIndex[matchParam.group(1)]
                daParams[iIdx] = float(matchParam.group(2))
    if np.any(np.isnan(daParams)):
        return None
    return daParams


# =========================================================================
# Corner Plot Construction
# =========================================================================


def ffigCreateCornerBase(daSamples, sColor):
    """Create a corner plot figure with one sampler's contours."""
    fig = corner.corner(
        daSamples,
        labels=listParamLabels,
        range=listBounds,
        color=sColor,
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        hist_kwargs={
            "density": True, "histtype": "step",
            "linewidth": 2.0, "color": sColor},
        contour_kwargs={"linewidths": 2.0, "colors": sColor},
        label_kwargs={"fontsize": D_LABEL_FONTSIZE},
        title_kwargs={"fontsize": 12},
        fig=plt.figure(figsize=(D_FIGSIZE, D_FIGSIZE)),
    )
    return fig


def fnOverlayCornerSamples(fig, daSamples, sColor):
    """Overlay a second sampler's contours onto an existing corner plot."""
    corner.corner(
        daSamples,
        fig=fig,
        range=listBounds,
        color=sColor,
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        hist_kwargs={
            "density": True, "histtype": "step",
            "linewidth": 2.0, "color": sColor},
        contour_kwargs={"linewidths": 2.0, "colors": sColor},
    )


def fnAddMaxLikelihoodPoints(fig, daMaxLikeParams):
    """Add ML vertical lines on diagonals and dots on off-diagonals."""
    axes = np.array(fig.axes).reshape((I_NUM_DIMENSIONS, I_NUM_DIMENSIONS))
    for i in range(I_NUM_DIMENSIONS):
        ax = axes[i, i]
        dYmin, dYmax = ax.get_ylim()
        ax.plot([daMaxLikeParams[i]] * 2, [0, dYmax],
                color="k", linewidth=2.0, alpha=0.8, zorder=10)
        for j in range(i):
            axes[i, j].plot(
                daMaxLikeParams[j], daMaxLikeParams[i], "ko",
                markersize=8, markeredgewidth=1.5, alpha=0.8, zorder=10)


def fnSetTickFontsize(fig):
    """Set tick label fontsize for all active panels in the corner plot."""
    axes = np.array(fig.axes).reshape((I_NUM_DIMENSIONS, I_NUM_DIMENSIONS))
    for i in range(I_NUM_DIMENSIONS):
        for j in range(I_NUM_DIMENSIONS):
            if i >= j:
                axes[i, j].tick_params(
                    axis="both", labelsize=D_TICK_FONTSIZE)


def fnAddPriorsToCorner(fig, kdeAgePrior):
    """Overlay prior distributions (grey dashed) on diagonal panels."""
    axes = np.array(fig.axes).reshape((I_NUM_DIMENSIONS, I_NUM_DIMENSIONS))
    for i in range(I_NUM_DIMENSIONS):
        ax = axes[i, i]
        dXmin, dXmax = listBounds[i]
        daXrange = np.linspace(dXmin, dXmax, 1000)
        daYprior = fdaPriorDensity(i, daXrange, kdeAgePrior)
        ax.plot(daXrange, daYprior, color="grey", linewidth=2.0,
                linestyle="--", alpha=0.7, zorder=0)


def fnAddLegendAndSave(fig, sOutputFile, listSamplerEntries,
                       bShowMaxLikelihood=True):
    """Add legend via figure coordinates, adjust spacing, and save."""
    fig.subplots_adjust(
        hspace=0.05, wspace=0.05,
        left=0.12, right=0.98, bottom=0.10, top=0.95)
    dLegendX = 0.72
    dLegendY = 0.87
    dLineLength = 0.03
    dTextOffset = 0.01
    dYspacing = 0.04
    listEntries = [(sColor, "-", 1.0, None, sLabel)
                   for sColor, sLabel in listSamplerEntries]
    listEntries.append(("grey", "--", 0.7, None, "Prior"))
    if bShowMaxLikelihood:
        listEntries.append(("k", None, 0.8, "o", "Max. Likelihood"))
    for iEntry, tEntry in enumerate(listEntries):
        fnDrawLegendEntry(fig, tEntry, dLegendX, dLegendY,
                          dLineLength, dTextOffset, dYspacing, iEntry)
    fig.savefig(sOutputFile, dpi=300)
    print(f"\n  Saved: {sOutputFile}")


def fnDrawLegendEntry(fig, tEntry, dLegendX, dLegendY,
                      dLineLength, dTextOffset, dYspacing, iEntry):
    """Draw a single legend entry on the figure."""
    sColor, sLinestyle, dAlpha, sMarker, sLabel = tEntry
    dY = dLegendY - iEntry * dYspacing
    if sMarker:
        fig.lines.append(plt.Line2D(
            [dLegendX + dLineLength / 2], [dY],
            marker=sMarker, color=sColor, markersize=8, linewidth=0,
            markeredgewidth=1.5, alpha=dAlpha, transform=fig.transFigure))
    else:
        fig.lines.append(plt.Line2D(
            [dLegendX, dLegendX + dLineLength], [dY, dY],
            color=sColor, linewidth=2, linestyle=sLinestyle,
            alpha=dAlpha, transform=fig.transFigure))
    fig.text(
        dLegendX + dLineLength + dTextOffset, dY, sLabel,
        fontsize=D_LEGEND_FONTSIZE, va="center", transform=fig.transFigure)


# =========================================================================
# Main
# =========================================================================


def main(sOutputFile):
    """Load sampler outputs and create the comparison corner plot."""
    sScriptDir = os.path.dirname(os.path.abspath(__file__))
    sSamplesDir = os.path.join(sScriptDir, "output")
    sAgeSamplesPath = str(
        Path(sScriptDir).parent / "EngleAgeDistribution" / "age_samples.txt")
    sMaxLevPath = str(
        Path(sScriptDir).parent / "MaximumLikelihood" / "maxlike_results.txt")

    kdeAgePrior = fkdeBuildAgePrior(sAgeSamplesPath, listBounds[3])

    daEmcee = fdaLoadSamplerSamples(sSamplesDir, "emcee_samples.npz")
    daDynesty = fdaLoadSamplerSamples(sSamplesDir, "dynesty_samples.npz")
    daMultinest = fdaLoadSamplerSamples(sSamplesDir, "multinest_samples.npz")
    daUltranest = fdaLoadSamplerSamples(sSamplesDir, "ultranest_samples.npz")

    sColorEmcee = vplot.colors.orange
    sColorDynesty = vplot.colors.pale_blue
    sColorMultinest = vplot.colors.purple
    sColorUltranest = vplot.colors.dark_blue

    listSamplerPlot = [
        (sColorEmcee, "Emcee", daEmcee),
        (sColorDynesty, "Dynesty", daDynesty),
        (sColorMultinest, "MultiNest", daMultinest),
        (sColorUltranest, "UltraNest", daUltranest),
    ]

    fig = ffigCreateCornerBase(listSamplerPlot[0][2], listSamplerPlot[0][0])
    for sColor, _, daSamples in listSamplerPlot[1:]:
        fnOverlayCornerSamples(fig, daSamples, sColor)

    fnSetTickFontsize(fig)

    daMaxLikeParams = fdaReadMaxLevParams(sMaxLevPath)
    if daMaxLikeParams is not None:
        fnAddMaxLikelihoodPoints(fig, daMaxLikeParams)

    fnAddPriorsToCorner(fig, kdeAgePrior)

    listLegendEntries = [(sColor, sLabel)
                         for sColor, sLabel, _ in listSamplerPlot]
    fnAddLegendAndSave(
        fig, sOutputFile, listLegendEntries,
        bShowMaxLikelihood=(daMaxLikeParams is not None))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sOutputFile = os.path.join("output", "sampler_comparison.pdf")
    else:
        sOutputFile = sys.argv[1]
    main(sOutputFile)
