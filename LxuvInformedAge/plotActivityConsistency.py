#!/usr/bin/env python3
"""
Plot GJ 1132's position within the activity-age relation's intrinsic scatter.

Shows the population prior on the offset z (standard normal, by construction
of the hierarchical model) against the posterior after conditioning on the
observed log(L_X/L_bol). A posterior consistent with the prior bulk means the
EMD framework passes this falsifiability check; the posterior shift measures
how XUV-quiet or -loud GJ 1132 is relative to coeval M dwarfs.

Usage: python plotActivityConsistency.py <outputPath>
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot  # noqa: F401  (applies the project figure style on import)

I_NUM_BINS = 60


def main():
    """Render the z prior-vs-posterior figure."""
    sOutputPath = sys.argv[1]
    daZ = np.loadtxt("zOffsetSamples.txt")
    with open("activityConsistency.json") as fileHandle:
        dictSummary = json.load(fileHandle)
    daGrid = np.linspace(-4, 4, 400)
    plt.figure(figsize=(6, 4.2))
    plt.plot(daGrid, np.exp(-0.5 * daGrid ** 2) / np.sqrt(2 * np.pi),
             color="0.4", linewidth=2, linestyle="--",
             label="population prior")
    plt.hist(daZ, bins=I_NUM_BINS, density=True, color="C0", alpha=0.6,
             label="posterior | observed $L_X$")
    plt.axvline(dictSummary["z_offset"]["mean"], color="C0", linewidth=1)
    plt.xlabel(r"offset $z$ within intrinsic scatter [$\sigma_{\rm int}$]")
    plt.ylabel("probability density")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
