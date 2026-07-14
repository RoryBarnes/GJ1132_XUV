#!/usr/bin/env python3
"""
What each observation of the host star buys for GJ 1132 b's XUV history.

(a) WHAT WE BELIEVE. The cumulative XUV flux with and without the host-star
X-ray measurement. The age prior is identical, so the gap between them is the
value of that measurement.

(b) WHAT A FUTURE SED WOULD DO. A panchromatic X-UV spectrum would sharpen the
answer -- but WHERE it lands is not known today. Each thin curve is one
possible future posterior: narrow, but centred wherever the measurement
happens to fall. Their mixture reproduces panel (a)'s X-ray posterior, as it
must. Most outcomes leave the planet well above the cosmic shoreline; a small
minority would leave the verdict genuinely ambiguous.

(c) WHERE THE UNCERTAINTY LIVES. Each state of knowledge as a stacked variance
budget. The stellar-population term collapses once the star is measured, and
the band conversion becomes the bottleneck.

The projected states assume the measured offsets PERSIST: the saturated phase
that sets the cumulative flux ended ~5 Gyr ago and cannot be observed.

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
I_NUM_FUTURE_CURVES = 25

S_COLOR_NONE = vplot.colors.orange
S_COLOR_XRAY = vplot.colors.dark_blue
S_COLOR_SED = vplot.colors.pale_blue
S_COLOR_FLOOR = vplot.colors.purple
S_COLOR_SHORELINE = vplot.colors.red

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
DICT_BUDGET_ORDER = {
    "noMeasurement": "no measurement",
    "xrayMeasurement": "X-ray measurement",
    "panchromaticSed": "*panchromatic SED",
    "relationFloor": "*relation floor",
}


def fdictLoadBudget():
    """Load the variance-budget product."""
    with open(os.path.join(S_DIRECTORY, "uncertaintyBudget.json")) as fileHandle:
        return json.load(fileHandle)


def fdaBins():
    """Return the common logarithmic flux bins."""
    return np.logspace(np.log10(40), np.log10(6000), I_NUM_BINS)


def fdaPeakNormalized(daFlux, daBins):
    """Return bin centres and a peak-normalized histogram.

    The states differ by up to 5.6x in width; peak-normalizing puts the WIDTHS
    -- what this figure is about -- on an equal footing. Absolute intervals are
    in the legend and in uncertaintyBudget.json.
    """
    daCounts, daEdges = np.histogram(daFlux, bins=daBins)
    return np.sqrt(daEdges[:-1] * daEdges[1:]), daCounts / max(daCounts.max(), 1)


def fnDrawShoreline(axis):
    """Mark the cosmic shoreline."""
    axis.axvline(D_SHORELINE_FLUX, color=S_COLOR_SHORELINE, linestyle=":",
                 linewidth=1.5, label="cosmic shoreline")


def fnPlotBelief(axis, dictBudget):
    """Panel (a): the two states of knowledge we can actually occupy."""
    daBins = fdaBins()
    for sKey, sColor in (("noMeasurement", S_COLOR_NONE),
                         ("xrayMeasurement", S_COLOR_XRAY)):
        dictScenario = dictBudget["dictScenarios"][sKey]
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY,
                                         f"fluxSamples_{sKey}.txt"))
        daCentres, daHeight = fdaPeakNormalized(daFlux, daBins)
        axis.fill_between(daCentres, daHeight, step="mid", color=sColor,
                          alpha=0.22, linewidth=0)
        axis.step(daCentres, daHeight, where="mid", color=sColor, linewidth=2,
                  label=f"{dictScenario['label']} "
                        f"($\\sigma$={dictScenario['sigma_log10']:.2f} dex)")
    fnDrawShoreline(axis)
    axis.set_xscale("log")
    axis.set_ylim(0, 1.35)
    axis.set_xlabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.set_ylabel("probability density (peak-normalized)")
    axis.set_title("(a) what we believe now", fontsize=9.5, loc="left")
    axis.legend(loc="upper left", fontsize=7.5)


def fdaFutureCurve(dCentre, dSigma, daBins):
    """Return one possible future posterior, peak-normalized on the flux axis."""
    daLogCentres = np.log10(np.sqrt(daBins[:-1] * daBins[1:]))
    daHeight = np.exp(-0.5 * ((daLogCentres - dCentre) / dSigma) ** 2)
    return 10 ** daLogCentres, daHeight


def fnPlotPreposterior(axis, dictBudget):
    """Panel (b): where a future panchromatic SED could actually land."""
    dictPre = dictBudget["dictPreposterior"]
    dSigmaFuture = dictBudget["dictScenarios"]["panchromaticSed"]["sigma_log10"]
    daBins = fdaBins()
    daCentres = np.array(dictPre["daPossibleCenters"])[:I_NUM_FUTURE_CURVES]
    for iIndex, dCentre in enumerate(daCentres):
        daX, daY = fdaFutureCurve(dCentre, dSigmaFuture, daBins)
        axis.plot(daX, daY, color=S_COLOR_SED, linewidth=1.0, alpha=0.55,
                  label="possible future posteriors" if iIndex == 0 else None)
    daFlux = np.loadtxt(os.path.join(S_DIRECTORY,
                                     "fluxSamples_xrayMeasurement.txt"))
    daX, daY = fdaPeakNormalized(daFlux, daBins)
    axis.step(daX, daY, where="mid", color=S_COLOR_XRAY, linewidth=2,
              linestyle="--", label="their mixture = today's posterior")
    fnDrawShoreline(axis)
    axis.set_xscale("log")
    axis.set_ylim(0, 1.35)
    axis.set_xlabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.set_title("(b) what a panchromatic SED would do", fontsize=9.5,
                   loc="left")
    axis.legend(loc="upper left", fontsize=7.5)
    axis.text(0.98, 0.55,
              f"centre uncertain by\n"
              f"$\\sigma$={dictPre['dSigmaCenterDex']:.2f} dex\n\n"
              f"P(still straddles\nthe shoreline) = "
              f"{100 * dictPre['dProbabilityStillStraddlesShoreline']:.1f}%",
              transform=axis.transAxes, ha="right", va="top", fontsize=7.5)


def fnPlotBudget(axis, dictBudget):
    """Panel (c): each state of knowledge as a stacked variance budget."""
    for iIndex, (sKey, sLabel) in enumerate(DICT_BUDGET_ORDER.items()):
        dictScenario = dictBudget["dictScenarios"][sKey]
        dLeft = 0.0
        for sSource, (sColor, sName) in DICT_SOURCE_STYLE.items():
            dVariance = dictScenario["dictVarianceBySource"].get(sSource, 0.0)
            axis.barh(iIndex, dVariance, left=dLeft, height=0.62, color=sColor,
                      label=sName if iIndex == 0 else None, edgecolor="white",
                      linewidth=0.5)
            dLeft += dVariance
        axis.text(dLeft + 0.004, iIndex,
                  f"$\\sigma$={dictScenario['sigma_log10']:.3f}",
                  va="center", fontsize=7, color="0.3")
    axis.set_yticks(range(len(DICT_BUDGET_ORDER)))
    axis.set_yticklabels(DICT_BUDGET_ORDER.values(), fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 0.215)
    axis.set_xlabel(r"variance of $\log_{10} F_{\rm XUV}$ [dex$^2$]")
    axis.set_title("(c) where the uncertainty lives", fontsize=9.5, loc="left")
    axis.legend(loc="lower right", fontsize=7)


def main():
    """Render the three-panel observational-value figure."""
    sOutputPath = sys.argv[1]
    dictBudget = fdictLoadBudget()
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, aaxis = plt.subplots(1, 3, figsize=(15, 4.4))
    fnPlotBelief(aaxis[0], dictBudget)
    fnPlotPreposterior(aaxis[1], dictBudget)
    fnPlotBudget(aaxis[2], dictBudget)
    figure.text(0.008, 0.005,
                "* Projected. Assumes the measured offset persists over the "
                "star's history: the saturated phase that sets the cumulative "
                "flux ended ~5 Gyr ago and cannot be observed.",
                fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
