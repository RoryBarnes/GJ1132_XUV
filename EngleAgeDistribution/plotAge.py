#!/usr/bin/env python3
"""
Plot GJ 1132's rotation-only age distribution under three treatments.

Blue and purple: the joint hierarchical refit, with every benchmark's latent age
marginalized, the full coefficient covariance and the fitted intrinsic-scatter
law, from its two independent samplers - dynesty nested sampling (age_samples.txt,
the distribution downstream steps use) and emcee (ageSamplesEmcee.txt).
Orange: the published M2.5-6.5 segmented fit of Engle & Guinan (2023) with
each coefficient drawn as an independent Gaussian of its quoted 1-sigma error
and no intrinsic scatter (ageSamplesIndependentGaussian.txt). All use the same
P_rot measurement distribution and a 13 Gyr ceiling. The histograms share bins
and are normalized to probability density so their shapes compare directly.

Usage: python plotAge.py <outputPath>
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
D_BIN_WIDTH_GYR = 0.1
D_MAX_AGE_GYR = 13.0
D_FIG_SIZE_X = 3.25
D_FIG_SIZE_Y = 3.0
D_HEADROOM = 1.12
LIST_DISTRIBUTIONS = [
    {"sFile": "age_samples.txt", "sLabel": "Joint refit, dynesty",
     "sColor": vplot.colors.dark_blue, "sStyle": "-"},
    {"sFile": "ageSamplesEmcee.txt", "sLabel": "Joint refit, emcee",
     "sColor": vplot.colors.purple, "sStyle": "-"},
    {"sFile": "ageSamplesIndependentGaussian.txt",
     "sLabel": "Published fit,\nindependent Gaussians",
     "sColor": vplot.colors.orange, "sStyle": "-"},
]


def fdaLoadAgesGyr(sFileName):
    """Load one age-sample file (years) from the step directory as Gyr."""
    sPath = os.path.join(S_DIRECTORY, sFileName)
    try:
        return np.loadtxt(sPath) / 1e9
    except OSError as error:
        sys.exit(f"ERROR: could not read {sPath}: {error}. Run dataAge.py first.")


def fdPlotDensity(axis, daAgeGyr, daBinEdges, dictStyle):
    """Draw one probability-density step histogram and return its peak."""
    daDensity, _ = np.histogram(daAgeGyr, bins=daBinEdges, density=True)
    axis.stairs(daDensity, daBinEdges, color=dictStyle["sColor"],
                linestyle=dictStyle["sStyle"], linewidth=1.4,
                label=dictStyle["sLabel"])
    return float(np.max(daDensity))


def fnPlotAgeDistributions(sOutputPath):
    """Overlay both age distributions on shared bins and save the figure."""
    daBinEdges = np.arange(0.0, D_MAX_AGE_GYR + D_BIN_WIDTH_GYR, D_BIN_WIDTH_GYR)
    figure, axis = plt.subplots(figsize=(D_FIG_SIZE_X, D_FIG_SIZE_Y))
    dPeak = max(fdPlotDensity(axis, fdaLoadAgesGyr(dictStyle["sFile"]),
                              daBinEdges, dictStyle)
                for dictStyle in LIST_DISTRIBUTIONS)
    axis.set_xlim(0, D_MAX_AGE_GYR)
    axis.set_ylim(0, dPeak * D_HEADROOM)
    axis.set_xlabel("Age [Gyr]", fontsize=12)
    axis.set_ylabel(r"Probability density [Gyr$^{-1}$]", fontsize=10)
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(sOutputPath, dpi=300)
    plt.close(figure)
    print(f"Histogram saved to: {sOutputPath}")


def main():
    """Render the age-distribution comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sOutputPath", help="Path of the figure to write.")
    fnPlotAgeDistributions(parser.parse_args().sOutputPath)


if __name__ == "__main__":
    main()
