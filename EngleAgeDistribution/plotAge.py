#!/usr/bin/env python3
"""
Plot Engle gyrochronology age distribution histogram from pre-computed samples.

Reads age_samples.txt (in years) and creates a normalized step histogram.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

I_NUM_BINS = 50
D_FIG_SIZE_X = 3.25
D_FIG_SIZE_Y = 3


def fnPlotNormalizedHistogram(daAgeGyr, sOutputPath):
    """Save a normalized histogram of the age distribution."""
    plt.figure(figsize=(D_FIG_SIZE_X, D_FIG_SIZE_Y))
    daCounts, daBinEdges = np.histogram(daAgeGyr, bins=I_NUM_BINS)
    daFractions = daCounts / len(daAgeGyr)
    plt.step(daBinEdges[:-1], daFractions, where='mid', color='k')
    plt.xlabel('Age [Gyr]', fontsize=12)
    plt.ylabel('Fraction', fontsize=12)
    plt.xlim(0, 13)
    plt.ylim(0, 0.04)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()
    print(f"Histogram saved to: {sOutputPath}")


if __name__ == "__main__":
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else "EngleAgeHist.pdf"

    daAgeYears = np.loadtxt('age_samples.txt')
    daAgeGyr = daAgeYears / 1e9
    fnPlotNormalizedHistogram(daAgeGyr, sOutputPath)
