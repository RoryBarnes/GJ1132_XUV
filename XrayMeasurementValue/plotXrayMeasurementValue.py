#!/usr/bin/env python3
"""
What each observation of the host star buys for GJ 1132 b's XUV history.

Left: the cumulative XUV flux under four states of knowledge, which differ ONLY
in what is known about the star -- the age prior is identical in all four, so
the gaps between them are the value of the observations. Solid: the two states
we can actually occupy (no measurement; the X-ray measurement we have). Thin:
two projections that assume the measured offsets PERSIST over the star's
history. The vconverge forward-model sweep is overlaid to show the analytic
propagation reproduces it.

Right: the same four states as stacked variance budgets. Each bar's length is
the total variance; its segments are the variance each source still
contributes. The population-scatter segment collapses when the star is
measured, and the band conversion becomes the bottleneck.

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

S_COLOR_SWEEP = "0.45"
S_COLOR_SHORELINE = vplot.colors.red

DICT_SCENARIO_STYLE = {
    "noMeasurement": {"sColor": vplot.colors.orange, "dWidth": 2.0,
                      "bFill": True},
    "xrayMeasurement": {"sColor": vplot.colors.dark_blue, "dWidth": 2.0,
                        "bFill": True},
    "panchromaticSed": {"sColor": vplot.colors.pale_blue, "dWidth": 1.6,
                        "bFill": False},
    "relationFloor": {"sColor": vplot.colors.purple, "dWidth": 1.4,
                      "bFill": False},
}
DICT_SOURCE_STYLE = {
    "stellar population (intrinsic scatter)":
        (vplot.colors.orange, "stellar population scatter"),
    "band conversion (intrinsic scatter)":
        (vplot.colors.pale_blue, "band-conversion scatter"),
    "X-UV relation coefficients (posterior covariance)":
        (vplot.colors.dark_blue, "X-UV relation covariance"),
    "band conversion (coefficient covariance)":
        (vplot.colors.purple, "band-conversion covariance"),
    "interaction residual": ("0.75", "interaction"),
}


def fdictLoadBudget():
    """Load the variance-budget product."""
    with open(os.path.join(S_DIRECTORY, "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def fsScenarioLabel(dictScenario, dictGain):
    """Build a legend label carrying the spread and the gain factor."""
    sMark = "*" if dictScenario["bProjected"] else ""
    return (f"{sMark}{dictScenario['label']} "
            f"($\\sigma$={dictScenario['sigma_log10']:.2f}, "
            f"{dictGain['dFactorVersusNoMeasurement']:.1f}x)")


def fdaPeakNormalized(daFlux, daBins):
    """Return bin centers and a peak-normalized histogram.

    The states of knowledge differ by up to 5.6x in width, so a common
    probability normalization would flatten the broad distributions into
    invisibility. Peak-normalizing puts the WIDTHS -- the quantity this
    figure is about -- on an equal footing. Absolute intervals are in the
    legend and in uncertaintyBudget.json.
    """
    daCounts, daEdges = np.histogram(daFlux, bins=daBins)
    return np.sqrt(daEdges[:-1] * daEdges[1:]), daCounts / daCounts.max()


def fnPlotDistribution(axis, daFlux, sLabel, dictStyle, daBins):
    """Overlay one peak-normalized flux histogram on a log axis."""
    daCenters, daHeight = fdaPeakNormalized(daFlux, daBins)
    if dictStyle["bFill"]:
        axis.fill_between(daCenters, daHeight, step="mid",
                          color=dictStyle["sColor"], alpha=0.22, linewidth=0)
    axis.step(daCenters, daHeight, where="mid", color=dictStyle["sColor"],
              linewidth=dictStyle["dWidth"], label=sLabel)


def fnPlotFluxPanel(axis, dictBudget):
    """Render the four states of knowledge as flux distributions."""
    daBins = np.logspace(np.log10(40), np.log10(6000), I_NUM_BINS)
    for sKey, dictStyle in DICT_SCENARIO_STYLE.items():
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY,
                                         f"fluxSamples_{sKey}.txt"))
        sLabel = fsScenarioLabel(dictBudget["dictScenarios"][sKey],
                                 dictBudget["dictGains"][sKey])
        fnPlotDistribution(axis, daFlux, sLabel, dictStyle, daBins)
    fnPlotSweepValidation(axis, daBins)
    axis.axvline(D_SHORELINE_FLUX, color=S_COLOR_SHORELINE, linestyle=":",
                 linewidth=1.5, label="cosmic shoreline")
    axis.set_xscale("log")
    axis.set_xlabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.set_ylabel("probability density (peak-normalized)")
    axis.set_ylim(0, 1.55)
    axis.legend(loc="upper left", fontsize=7, framealpha=0.9)


def fnPlotSweepValidation(axis, daBins):
    """Overlay the vconverge sweep that validates the analytic propagation."""
    daSweep = np.loadtxt(os.path.join(S_DIRECTORY,
                                      "fluxSamplesForwardModelSweep.txt"))
    daCenters, daHeight = fdaPeakNormalized(daSweep, daBins)
    axis.step(daCenters, daHeight, where="mid", color=S_COLOR_SWEEP,
              linewidth=1.2, linestyle="--", label="forward-model sweep")


def fnPlotVarianceSegment(axis, dY, dLeft, dVariance, sSource, bLabel):
    """Draw one source's variance segment in a stacked bar."""
    sColor, sLabel = DICT_SOURCE_STYLE[sSource]
    axis.barh(dY, dVariance, left=dLeft, height=0.62, color=sColor,
              label=sLabel if bLabel else None, edgecolor="white",
              linewidth=0.5)


def fnPlotBudgetPanel(axis, dictBudget):
    """Render each state of knowledge as a stacked variance budget."""
    saKeys = list(DICT_SCENARIO_STYLE.keys())
    for iIndex, sKey in enumerate(saKeys):
        dictScenario = dictBudget["dictScenarios"][sKey]
        dLeft = 0.0
        for sSource in DICT_SOURCE_STYLE:
            dVariance = dictScenario["dictVarianceBySource"].get(sSource, 0.0)
            fnPlotVarianceSegment(axis, iIndex, dLeft, dVariance, sSource,
                                  iIndex == 0)
            dLeft += dVariance
        axis.text(dLeft + 0.004, iIndex,
                  f"$\\sigma$={dictScenario['sigma_log10']:.3f}",
                  va="center", fontsize=7.5, color="0.3")
    axis.set_yticks(range(len(saKeys)))
    axis.set_yticklabels(
        [("*" if dictBudget["dictScenarios"][k]["bProjected"] else "")
         + dictBudget["dictScenarios"][k]["label"].replace(" (", "\n(")
         for k in saKeys], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(r"variance of $\log_{10} F_{\rm XUV}$ [dex$^2$]")
    axis.set_xlim(0, 0.215)
    axis.legend(loc="lower right", fontsize=7.5)


def main():
    """Render the two-panel observational-value figure."""
    sOutputPath = sys.argv[1]
    dictBudget = fdictLoadBudget()
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, (axisLeft, axisRight) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fnPlotFluxPanel(axisLeft, dictBudget)
    fnPlotBudgetPanel(axisRight, dictBudget)
    figure.text(0.01, 0.005,
                "* Projected. Assumes the measured offset persists over the "
                "star's history: the saturated phase that sets the cumulative "
                "flux ended ~5 Gyr ago and cannot be observed.",
                fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
