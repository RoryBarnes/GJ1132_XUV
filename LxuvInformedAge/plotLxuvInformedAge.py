#!/usr/bin/env python3
"""
Plot GJ 1132's rotation-only, L_XUV-only, and L_XUV-informed age posteriors.

Shows how folding the observed L_XUV/L_bol into the age inference reshapes the
rotation-only age PDF. The breadth of the L_XUV-only curve reflects the large
intrinsic scatter of the X-UV activity-age relation inferred in Stage 1.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

I_NUM_BINS = 60


def fnPlotAgeHistogram(daAgeGyr, sLabel, sColor, dAlpha):
    """Overlay one normalized step histogram of an age distribution."""
    daCounts, daEdges = np.histogram(daAgeGyr, bins=I_NUM_BINS, range=(0, 13))
    daFractions = daCounts / len(daAgeGyr)
    daCenters = 0.5 * (daEdges[:-1] + daEdges[1:])
    plt.step(daCenters, daFractions, where="mid", color=sColor, alpha=dAlpha,
             linewidth=2, label=sLabel)


def fnFormatAxes():
    """Apply axis labels and limits to the current figure."""
    plt.xlabel("Age [Gyr]", fontsize=12)
    plt.ylabel("Fraction", fontsize=12)
    plt.xlim(0, 13)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=9)


def main():
    """Generate the three-way age-posterior comparison figure."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else "LxuvInformedAge.pdf"
    daRotation = np.loadtxt("rotationOnlyAgeSamples.txt") / 1e9
    daInformed = np.loadtxt("lxuvInformedAgeSamples.txt") / 1e9
    plt.figure(figsize=(5, 3.5))
    fnPlotAgeHistogram(daRotation, "Rotation only", "k", 1.0)
    fnPlotAgeHistogram(daInformed, "L$_{\\rm XUV}$-informed",
                       vplot.colors.orange, 1.0)
    fnFormatAxes()
    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()
    print(f"Age comparison figure saved to: {sOutputPath}")


if __name__ == "__main__":
    main()
