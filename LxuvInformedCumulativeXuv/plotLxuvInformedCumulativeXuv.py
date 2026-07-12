#!/usr/bin/env python3
"""
Plot the cumulative XUV flux on GJ 1132 b for two EMD age priors.

Compares the rotation-only and L_XUV-informed cumulative-flux distributions on a
common log axis, with the published EMD-only result and the cosmic shoreline
marked for reference. Isolating the two priors under one propagation shows the
effect of folding the observed L_XUV/L_bol into the age.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (D_LOWER_BOUND, D_UPPER_BOUND, D_SHORELINE_FLUX,
                                 ftComputeLogBins, ftComputeStatistics)

D_PUBLISHED_MEAN = 402.0
S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def fnPlotFluxHistogram(daFlux, sLabel, sColor):
    """Overlay one normalized log-binned cumulative-flux histogram."""
    daBins, daFractions = ftComputeLogBins(daFlux)
    plt.step(daBins, daFractions, where="mid", color=sColor, linewidth=2,
             label=sLabel)


D_PUBLISHED_FLARES_MEAN = 484.0


def fnDrawReferenceLines():
    """Draw the cosmic shoreline and published EMD means for context."""
    plt.axvline(D_SHORELINE_FLUX, color=vplot.colors.pale_blue, linewidth=6)
    plt.axvline(D_PUBLISHED_MEAN, color="k", linestyle="--", linewidth=1.2,
                label="Published EMD-only (402)")
    plt.axvline(D_PUBLISHED_FLARES_MEAN, color="k", linestyle=":", linewidth=1.2,
                label="Published EMD+flares (484)")
    plt.annotate("Cosmic Shoreline", (D_SHORELINE_FLUX * 0.8, 0.005),
                 fontsize=10, rotation=90, color=vplot.colors.pale_blue)


def fnFormatAxes():
    """Apply log-axis formatting to the cumulative-flux figure."""
    plt.xlabel("Normalized Cumulative XUV Flux", fontsize=13)
    plt.ylabel("Fraction", fontsize=13)
    plt.xscale("log")
    plt.xlim(D_LOWER_BOUND, D_UPPER_BOUND)
    plt.legend(loc="upper left", fontsize=9)


def fnPrintStatistics(daFlux, sLabel):
    """Print mean and 95% CI for a flux distribution."""
    dMean, dLower, dUpper = ftComputeStatistics(daFlux)
    print(f"{sLabel}: mean {dMean:.0f}, 95% CI [{dLower:.0f}, {dUpper:.0f}]")


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot L_XUV-informed vs rotation-only cumulative XUV flux.")
    parser.add_argument("output_path", help="Destination for the figure.")
    return parser.parse_args()


def main():
    """Generate the cumulative-flux comparison figure."""
    args = ftParseArguments()
    dictSeries = flistLoadSeries()
    plt.figure(figsize=(6.5, 5))
    fnDrawReferenceLines()
    for sLabel, (sFile, sColor) in dictSeries.items():
        daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sFile))
        fnPlotFluxHistogram(daFlux, sLabel, sColor)
        fnPrintStatistics(daFlux, sLabel)
    fnFormatAxes()
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300)
    plt.close()


def flistLoadSeries():
    """Return the mapping of legend label to (sample file, colour)."""
    return {
        "Rotation only": ("cumulativeXuvFluxSamplesRotationOnly.txt", "grey"),
        r"L$_{\rm XUV}$-informed": ("cumulativeXuvFluxSamplesInformed.txt",
                                    vplot.colors.orange),
        "Rotation + flares": ("cumulativeXuvFluxSamplesFlaresRotationOnly.txt",
                              vplot.colors.dark_blue),
        r"L$_{\rm XUV}$-informed + flares":
            ("cumulativeXuvFluxSamplesFlaresInformed.txt", vplot.colors.red),
    }


if __name__ == "__main__":
    main()
