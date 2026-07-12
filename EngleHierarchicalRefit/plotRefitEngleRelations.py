#!/usr/bin/env python3
"""
Plot the hierarchical refit of the two Engle EMD relations.

Left panel: age-rotation relation with the M4-6.5 benchmarks. Right panel: the
X-UV(5-1700A)/L_bol activity-age relation with the reconstructed mid-late M
sample. Each panel shows the data with error bars, the posterior-median hinge,
and the inferred intrinsic-scatter band.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataRefitEngleRelations import (
    fdaHinge, ftLoadRotationBenchmarks, ftBuildRotationLogAgeErrors,
    ftLoadXuvActivityData)

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def fdaMedianCoefficients(sSamplesFile):
    """Return posterior-median [a, b, c, d, sig_young, sig_old]."""
    daChain = np.load(os.path.join(S_DIRECTORY, sSamplesFile))
    daMedian = np.median(daChain, axis=0)
    daMedian[4] = np.exp(daMedian[4])
    daMedian[5] = np.exp(daMedian[5])
    return daMedian


def fnDrawRelation(axis, daX, daModel, daSigmaYoung, daSigmaOld, dBreak):
    """Draw the median hinge and its segment-dependent scatter band."""
    daSigma = np.where(daX >= dBreak, daSigmaOld, daSigmaYoung)
    axis.plot(daX, daModel, color=vplot.colors.dark_blue, linewidth=2,
              label="Refit median")
    axis.fill_between(daX, daModel - daSigma, daModel + daSigma,
                      color=vplot.colors.pale_blue, alpha=0.5,
                      label=r"$\pm\,\sigma_{\rm int}$")


def fnPlotRotationPanel(axis):
    """Render the age-rotation relation panel."""
    daProt, daAge, daPlus, daMinus = ftLoadRotationBenchmarks()
    daTau, daSigPlus, daSigMinus = ftBuildRotationLogAgeErrors(
        daAge, daPlus, daMinus)
    daMedian = fdaMedianCoefficients("rotationCoefficientSamples.npy")
    axis.errorbar(daProt, daTau, yerr=[daSigMinus, daSigPlus], fmt="o",
                  color="k", markersize=4, elinewidth=1, capsize=2,
                  label="M4-6.5 benchmarks")
    daGrid = np.linspace(daProt.min(), daProt.max(), 300)
    fnDrawRelation(axis, daGrid, fdaHinge(daMedian[:4], daGrid),
                   daMedian[4], daMedian[5], daMedian[3])
    axis.set_xlabel(r"$P_{\rm rot}$ [days]", fontsize=12)
    axis.set_ylabel(r"$\log_{10}(\mathrm{Age/Gyr})$", fontsize=12)
    axis.legend(fontsize=8)


def fnPlotXuvPanel(axis):
    """Render the X-UV activity-age relation panel."""
    daTau, daTauErr, daY, daYErr = ftLoadXuvActivityData()
    daMedian = fdaMedianCoefficients("xuvCoefficientSamples.npy")
    axis.errorbar(daTau, daY, xerr=daTauErr, yerr=daYErr, fmt="o", color="k",
                  markersize=3, elinewidth=0.8, capsize=1.5,
                  label="Mid-late M (X-UV)")
    daGrid = np.linspace(daTau.min(), daTau.max(), 300)
    fnDrawRelation(axis, daGrid, fdaHinge(daMedian[:4], daGrid),
                   daMedian[4], daMedian[5], daMedian[3])
    axis.set_xlabel(r"$\log_{10}(\mathrm{Age/Gyr})$", fontsize=12)
    axis.set_ylabel(r"$\log_{10}(L_{\rm XUV}/L_{\rm bol})$", fontsize=12)
    axis.legend(fontsize=8)


def main():
    """Generate the two-panel refit figure."""
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else "RefitEngleRelations.pdf"
    figure, tAxes = plt.subplots(1, 2, figsize=(9, 4))
    fnPlotRotationPanel(tAxes[0])
    fnPlotXuvPanel(tAxes[1])
    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()
    print(f"Refit figure saved to: {sOutputPath}")


if __name__ == "__main__":
    main()
