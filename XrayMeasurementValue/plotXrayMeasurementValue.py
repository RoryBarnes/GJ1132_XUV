#!/usr/bin/env python3
"""
Show what one X-ray measurement of the host star buys.

Left: the cumulative XUV flux GJ 1132 b received, with and without a host-star
X-ray measurement. The two differ ONLY in the star's offset within the activity
relation's population scatter -- the age prior is identical -- so the gap
between them is the value of the measurement. The vconverge forward-model sweep
is overlaid to show the analytic propagation reproduces it.

Right: first-order variance indices of the four uncertainty sources. Without
the measurement the stellar population scatter dominates; the measurement
collapses that term and leaves the band conversion as the new floor.

Usage: python plotXrayMeasurementValue.py <outputPath>
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import D_SHORELINE_FLUX

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
I_NUM_BINS = 55

S_COLOR_POPULATION = vplot.colors.orange
S_COLOR_INFORMED = vplot.colors.dark_blue
S_COLOR_SWEEP = vplot.colors.purple
S_COLOR_SHORELINE = vplot.colors.red

DICT_SOURCE_LABELS = {
    "stellar population (intrinsic scatter)":
        "stellar population\n" r"($\sigma_{\rm int}$ at GJ 1132's age)",
    "band conversion (intrinsic scatter)":
        "band conversion\n(SED-to-SED scatter)",
    "X-UV relation coefficients (posterior covariance)":
        "X-UV relation\n(coefficient covariance)",
    "band conversion (coefficient covariance)":
        "band conversion\n(coefficient covariance)",
}


def fdictLoadBudget():
    """Load the variance-budget product."""
    with open(os.path.join(S_DIRECTORY, "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def fnPlotDistribution(axis, daFlux, sLabel, sColor, daBins):
    """Overlay one normalized, filled flux histogram on a log axis."""
    daCounts, daEdges = np.histogram(daFlux, bins=daBins)
    daCenters = np.sqrt(daEdges[:-1] * daEdges[1:])
    daFraction = daCounts / len(daFlux)
    axis.fill_between(daCenters, daFraction, step="mid", color=sColor,
                      alpha=0.25, linewidth=0)
    axis.step(daCenters, daFraction, where="mid", color=sColor, linewidth=2,
              label=sLabel)


def fnPlotSweepValidation(axis, daSweep, daBins):
    """Overlay the vconverge sweep that validates the analytic propagation."""
    daCounts, daEdges = np.histogram(daSweep, bins=daBins)
    daCenters = np.sqrt(daEdges[:-1] * daEdges[1:])
    axis.step(daCenters, daCounts / len(daSweep), where="mid",
              color=S_COLOR_SWEEP, linewidth=1.3, linestyle="--",
              label="forward-model sweep")


def fnAnnotateInterval(axis, dictStats, dY, sColor):
    """Draw one distribution's 95% interval and median."""
    axis.plot(dictStats["ci95"], [dY, dY], color=sColor, linewidth=2.5,
              solid_capstyle="butt")
    axis.plot([dictStats["median"]], [dY], marker="|", color=sColor,
              markersize=12, markeredgewidth=2.5)


def fnAnnotateGain(axis, dictBudget):
    """State the measurement's value directly on the figure."""
    dictP, dictI = dictBudget["dictPopulationZ"], dictBudget["dictInformedZ"]
    dSpanPopulation = dictP["ci95"][1] / dictP["ci95"][0]
    dSpanInformed = dictI["ci95"][1] / dictI["ci95"][0]
    axis.text(0.97, 0.60,
              f"95% interval spans\n{dSpanPopulation:.0f}x without\n"
              f"{dSpanInformed:.0f}x with\n"
              r"$\Rightarrow$ " f"{dSpanPopulation / dSpanInformed:.1f}x tighter",
              transform=axis.transAxes, ha="right", va="top", fontsize=8.5)


def fnPlotFluxPanel(axis, dictBudget):
    """Render the with/without-measurement flux comparison."""
    daBins = np.logspace(np.log10(30), np.log10(6000), I_NUM_BINS)
    for sFile, sLabel, sColor in (
            ("fluxSamplesPopulationZ.txt", "no X-ray measurement",
             S_COLOR_POPULATION),
            ("fluxSamplesInformedZ.txt", "with X-ray measurement",
             S_COLOR_INFORMED)):
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sFile))
        fnPlotDistribution(axis, daFlux, sLabel, sColor, daBins)
    fnPlotSweepValidation(axis, np.loadtxt(os.path.join(
        S_DIRECTORY, "fluxSamplesForwardModelSweep.txt")), daBins)
    fnAnnotateInterval(axis, dictBudget["dictPopulationZ"], 0.056,
                       S_COLOR_POPULATION)
    fnAnnotateInterval(axis, dictBudget["dictInformedZ"], 0.061,
                       S_COLOR_INFORMED)
    axis.axvline(D_SHORELINE_FLUX, color=S_COLOR_SHORELINE, linestyle=":",
                 linewidth=1.5, label="cosmic shoreline")
    axis.set_xscale("log")
    axis.set_xlabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.set_ylabel("fraction of realizations")
    axis.set_ylim(0, 0.068)
    axis.legend(loc="upper left", fontsize=8)
    fnAnnotateGain(axis, dictBudget)


def fdaSharesInOrder(dictLevel2, saKeys):
    """Return the first-order variance shares for the given sources, in order."""
    dictShares = dictLevel2["dictSourceShares"]
    return np.array([dictShares[sKey]["dVarianceShare"] for sKey in saKeys])


def fnAnnotateForwardModel(axis, dictLevel1):
    """Record the level-1 result: the X-UV block carries nearly all variance."""
    dBlock = dictLevel1["dictBlockShares"]["X-UV relation coefficients"]
    axis.text(0.98, 0.02,
              f"forward model ({dictLevel1['iTrials']} trials, "
              f"$R^2$={dictLevel1['dTotalRSquared']:.3f}):\n"
              f"X-UV relation carries {100 * dBlock:.1f}% of the variance;\n"
              "stellar age, mass and planet parameters <0.5% each",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5,
              color="0.35")


def fnPlotBudgetPanel(axis, dictBudget):
    """Render the first-order variance indices as grouped bars."""
    saKeys = list(DICT_SOURCE_LABELS.keys())
    daPopulation = fdaSharesInOrder(dictBudget["dictLevel2PopulationZ"], saKeys)
    daInformed = fdaSharesInOrder(dictBudget["dictLevel2InformedZ"], saKeys)
    dSigmaPopulation = dictBudget["dictPopulationZ"]["sigma_log10"]
    dSigmaInformed = dictBudget["dictInformedZ"]["sigma_log10"]
    daY = np.arange(len(saKeys))
    axis.barh(daY + 0.19, 100 * daPopulation, height=0.36,
              color=S_COLOR_POPULATION,
              label=f"no X-ray meas. ($\\sigma$={dSigmaPopulation:.2f} dex)")
    axis.barh(daY - 0.19, 100 * daInformed, height=0.36,
              color=S_COLOR_INFORMED,
              label=f"with X-ray meas. ($\\sigma$={dSigmaInformed:.2f} dex)")
    axis.set_yticks(daY)
    axis.set_yticklabels([DICT_SOURCE_LABELS[sKey] for sKey in saKeys],
                         fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("first-order variance index [%]")
    axis.set_xlim(0, 74)
    axis.legend(loc="center right", fontsize=8)
    fnAnnotateForwardModel(axis, dictBudget["dictLevel1ForwardModel"])


def main():
    """Render the two-panel X-ray-measurement-value figure."""
    sOutputPath = sys.argv[1]
    dictBudget = fdictLoadBudget()
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, (axisLeft, axisRight) = plt.subplots(1, 2, figsize=(11, 4.3))
    fnPlotFluxPanel(axisLeft, dictBudget)
    fnPlotBudgetPanel(axisRight, dictBudget)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
