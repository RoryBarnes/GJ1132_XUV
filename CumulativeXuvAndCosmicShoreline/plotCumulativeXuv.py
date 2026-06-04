"""Plot cumulative XUV histograms for the four model variants."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import vplot

from utils.cumulativeXuv import (D_LOWER_BOUND, D_UPPER_BOUND,
                                  D_SHORELINE_FLUX, ftGatherFluxes)


SA_LABELS = ["Engle Only", "Engle w/Flares", "Ribas Only", "Ribas w/Flares"]
SA_COLORS = ['grey', 'k', vplot.colors.orange, vplot.colors.orange]
DA_ALPHAS = [1.0, 1.0, 0.5, 1.0]

S_OWN_ENGLE_CONVERGED = "Engle/output/engle_converged.json"
S_OWN_RIBAS_CONVERGED = "Ribas/output/ribas_converged.json"


def fnPlotHistograms(listData):
    """Plot step histograms for all model variants."""
    plt.figure(figsize=(6.5, 6))
    plt.axvline(D_SHORELINE_FLUX, color=vplot.colors.pale_blue, linewidth=6)
    for i, (daBins, daFractions, _, _, _) in enumerate(listData):
        plt.step(daBins, daFractions, where='mid', color=SA_COLORS[i],
                 alpha=DA_ALPHAS[i], linestyle='-', linewidth=2,
                 label=SA_LABELS[i])


def fnFormatAxes():
    """Apply axis formatting to the current figure."""
    plt.xlabel('Normalized Cumulative XUV Flux', fontsize=24)
    plt.ylabel('Fraction', fontsize=24)
    plt.xlim(D_LOWER_BOUND, D_UPPER_BOUND)
    plt.xscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 0.25)
    plt.legend(loc='upper left', fontsize=16, framealpha=0.9)
    plt.annotate('Cosmic Shoreline', (40, 0.06), fontsize=20,
                 rotation=90, color=vplot.colors.pale_blue)


def fnPrintStatistics(listData):
    """Print mean and confidence interval for each variant."""
    for i, sLabel in enumerate(SA_LABELS):
        dMean, dLower, dUpper = listData[i][2], listData[i][3], listData[i][4]
        print(f"{sLabel} - Mean: {dMean:.2f}, "
              f"95% CI: [{dLower:.2f}, {dUpper:.2f}]")


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot cumulative XUV histograms for four model variants."
    )
    parser.add_argument("output_path",
                        help="Destination path for the histogram figure.")
    parser.add_argument("--engle-barnes-converged", required=True,
                        help="Path to A09 EngleBarnes converged JSON.")
    parser.add_argument("--ribas-barnes-converged", required=True,
                        help="Path to A09 RibasBarnes converged JSON.")
    return parser.parse_args()


def flistLoadAllVariants(args, sScriptDirectory):
    """Return the list of per-variant gather-flux results."""
    return [
        ftGatherFluxes(os.path.join(sScriptDirectory, S_OWN_ENGLE_CONVERGED)),
        ftGatherFluxes(args.engle_barnes_converged),
        ftGatherFluxes(os.path.join(sScriptDirectory, S_OWN_RIBAS_CONVERGED)),
        ftGatherFluxes(args.ribas_barnes_converged),
    ]


def main():
    """Generate cumulative XUV flux histogram for all model variants."""
    args = ftParseArguments()
    sScriptDirectory = os.path.dirname(os.path.abspath(__file__))
    listData = flistLoadAllVariants(args, sScriptDirectory)
    fnPlotHistograms(listData)
    fnFormatAxes()
    plt.savefig(args.output_path, dpi=300)
    plt.close()
    fnPrintStatistics(listData)


if __name__ == "__main__":
    main()
