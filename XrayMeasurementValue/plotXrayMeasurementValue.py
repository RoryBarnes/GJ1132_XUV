#!/usr/bin/env python3
"""
Show what one X-ray measurement of the host star buys.

Left: the cumulative XUV flux GJ 1132 b received, with and without a host-star
X-ray measurement. The two differ ONLY in the star's offset z within the
activity relation's population scatter -- the age prior is identical -- so the
gap between them is the value of the measurement. The forward-model sweep
(population z) is overlaid to show the analytic propagation reproduces it.

Right: the uncertainty budget. Without the measurement, the population scatter
dominates; the measurement collapses that one term and leaves the band
conversion as the new floor.

Usage: python plotXrayMeasurementValue.py <outputPath>
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot  # noqa: F401  (applies the project figure style on import)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (fdictLoadConvergedJson, daExtractFluxValues,
                                 D_SHORELINE_FLUX)

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
I_NUM_BINS = 55

SA_COMPONENT_LABELS = ["refit\ncoefficients", "conversion\ncoefficients",
                       "conversion\nscatter", "population\nscatter"]


def fdictLoadBudget():
    """Load the uncertainty-budget product."""
    with open(os.path.join(S_DIRECTORY, "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def fnPlotDistribution(axis, daFlux, sLabel, sColor, daBins, bFill):
    """Overlay one normalized flux histogram on a log axis."""
    daCounts, daEdges = np.histogram(daFlux, bins=daBins)
    daCenters = np.sqrt(daEdges[:-1] * daEdges[1:])
    daFraction = daCounts / len(daFlux)
    if bFill:
        axis.fill_between(daCenters, daFraction, step="mid", color=sColor,
                          alpha=0.25, linewidth=0)
    axis.step(daCenters, daFraction, where="mid", color=sColor, linewidth=2,
              label=sLabel)


def fnPlotSweepValidation(axis, daSweep, daBins):
    """Overlay the vconverge forward-model sweep that validates the analytics."""
    daCounts, daEdges = np.histogram(daSweep, bins=daBins)
    daCenters = np.sqrt(daEdges[:-1] * daEdges[1:])
    axis.step(daCenters, daCounts / len(daSweep), where="mid", color="0.25",
              linewidth=1.2, linestyle="--",
              label="vconverge sweep (validation)")


def fnAnnotateInterval(axis, dictStats, dY, sColor):
    """Draw the 95% interval and median of one distribution."""
    axis.plot(dictStats["ci95"], [dY, dY], color=sColor, linewidth=2.5,
              solid_capstyle="butt")
    axis.plot([dictStats["median"]], [dY], marker="|", color=sColor,
              markersize=12, markeredgewidth=2.5)


def fnPlotFluxPanel(axis, dictBudget):
    """Render the with/without-measurement flux comparison."""
    daPopulation = np.loadtxt(os.path.join(S_DIRECTORY,
                                           "fluxSamplesPopulationZ.txt"))
    daInformed = np.loadtxt(os.path.join(S_DIRECTORY,
                                         "fluxSamplesInformedZ.txt"))
    daSweep = np.loadtxt(os.path.join(S_DIRECTORY,
                                      "fluxSamplesForwardModelSweep.txt"))
    daBins = np.logspace(np.log10(30), np.log10(6000), I_NUM_BINS)
    fnPlotDistribution(axis, daPopulation, "no X-ray measurement", "C1",
                       daBins, True)
    fnPlotDistribution(axis, daInformed, "with X-ray measurement", "C0",
                       daBins, True)
    fnPlotSweepValidation(axis, daSweep, daBins)
    fnAnnotateInterval(axis, dictBudget["dictPopulationZ"], 0.056, "C1")
    fnAnnotateInterval(axis, dictBudget["dictInformedZ"], 0.061, "C0")
    axis.axvline(D_SHORELINE_FLUX, color="0.35", linestyle=":", linewidth=1.5,
                 label="cosmic shoreline")
    axis.set_xscale("log")
    axis.set_xlabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.set_ylabel("fraction of realizations")
    axis.set_ylim(0, 0.068)
    axis.legend(loc="upper left", fontsize=8)
    fnAnnotateGain(axis, dictBudget)


def fnAnnotateGain(axis, dictBudget):
    """State the measurement's value directly on the figure."""
    dictP, dictI = dictBudget["dictPopulationZ"], dictBudget["dictInformedZ"]
    dWidthPopulation = dictP["ci95"][1] / dictP["ci95"][0]
    dWidthInformed = dictI["ci95"][1] / dictI["ci95"][0]
    axis.text(0.97, 0.62,
              f"95% interval spans\n"
              f"{dWidthPopulation:.0f}x without\n"
              f"{dWidthInformed:.0f}x with\n"
              r"$\Rightarrow$ " + f"{dWidthPopulation / dWidthInformed:.1f}x "
              f"tighter",
              transform=axis.transAxes, ha="right", va="top", fontsize=8.5)


def fdaComponentSpreads(dictComponents, sMode):
    """Return the four component spreads for one state of knowledge."""
    sKey = f"D_population_scatter_{sMode}_z"
    return np.array([dictComponents["A_refit_mean_line"],
                     dictComponents["B_conversion_covariance"],
                     dictComponents["C_conversion_scatter"],
                     dictComponents[sKey]])


def fnPlotBudgetPanel(axis, dictBudget):
    """Render the uncertainty budget as grouped bars."""
    dictComponents = dictBudget["dictComponentsDex"]
    daPopulation = fdaComponentSpreads(dictComponents, "population")
    daInformed = fdaComponentSpreads(dictComponents, "informed")
    daY = np.arange(len(SA_COMPONENT_LABELS))
    axis.barh(daY + 0.19, daPopulation, height=0.36, color="C1",
              label="no X-ray measurement")
    axis.barh(daY - 0.19, daInformed, height=0.36, color="C0",
              label="with X-ray measurement")
    axis.set_yticks(daY)
    axis.set_yticklabels(SA_COMPONENT_LABELS, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(r"contribution to $\sigma[\log_{10} F_{\rm XUV}]$ [dex]")
    axis.legend(loc="upper right", fontsize=8)


def main():
    """Render the two-panel X-ray-measurement-value figure."""
    sOutputPath = sys.argv[1]
    dictBudget = fdictLoadBudget()
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, (axisLeft, axisRight) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fnPlotFluxPanel(axisLeft, dictBudget)
    fnPlotBudgetPanel(axisRight, dictBudget)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
