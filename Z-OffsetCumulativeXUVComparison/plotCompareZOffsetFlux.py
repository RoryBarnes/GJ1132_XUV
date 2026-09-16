#!/usr/bin/env python3
"""
Plot cumulative XUV flux under the population vs informed scatter-offset priors.

One panel: the four flux distributions (quiescent and flare-inclusive, each
under z ~ population prior and z ~ informed posterior) on a common log axis
with the cosmic shoreline marked. The population/informed separation is the
value of a single host-star X-ray measurement; the quiescent/flare separation
is the flare contribution.

Usage: python plotCompareZOffsetFlux.py <outputPath>
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot  # noqa: F401  (applies the project figure style on import)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import D_SHORELINE_FLUX

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
I_NUM_BINS = 40

LIST_SERIES = [
    ("cumulativeXuvFluxSamplesZPopulation.txt", "quiescent, population z",
     "C0", "-"),
    ("cumulativeXuvFluxSamplesZInformed.txt", "quiescent, informed z",
     "C0", "--"),
    ("cumulativeXuvFluxSamplesFlaresZPopulation.txt", "with flares, population z",
     "C1", "-"),
    ("cumulativeXuvFluxSamplesFlaresZInformed.txt", "with flares, informed z",
     "C1", "--"),
]


def fnPlotOneSeries(sFilename, sLabel, sColor, sLinestyle, daBins):
    """Overlay one normalized step histogram of a flux distribution."""
    daFlux = np.loadtxt(os.path.join(S_DIRECTORY, sFilename))
    daCounts, daEdges = np.histogram(daFlux, bins=daBins)
    daCenters = np.sqrt(daEdges[:-1] * daEdges[1:])
    plt.step(daCenters, daCounts / len(daFlux), where="mid", color=sColor,
             linestyle=sLinestyle, linewidth=2, label=sLabel)


def main():
    """Render the four-way flux comparison figure."""
    sOutputPath = sys.argv[1]
    daAll = np.concatenate([np.loadtxt(os.path.join(S_DIRECTORY, t[0]))
                            for t in LIST_SERIES])
    daBins = np.logspace(np.log10(daAll.min() * 0.9),
                         np.log10(daAll.max() * 1.1), I_NUM_BINS)
    plt.figure(figsize=(7, 4.5))
    for sFilename, sLabel, sColor, sLinestyle in LIST_SERIES:
        fnPlotOneSeries(sFilename, sLabel, sColor, sLinestyle, daBins)
    plt.axvline(D_SHORELINE_FLUX, color="0.4", linestyle=":",
                label="cosmic shoreline")
    plt.xscale("log")
    plt.xlabel("cumulative XUV flux [relative to Earth]")
    plt.ylabel("fraction of realizations")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
