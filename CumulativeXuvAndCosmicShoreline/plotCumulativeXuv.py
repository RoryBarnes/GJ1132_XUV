import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import vplot

from utils.cumulativeXuv import (D_LOWER_BOUND, D_UPPER_BOUND,
                                  D_SHORELINE_FLUX, ftGatherFluxes)


SA_DIRECTORIES = ['Engle', 'EngleBarnes', 'Ribas', 'RibasBarnes']
SA_LABELS = ["Engle Only", "Engle w/Flares", "Ribas Only", "Ribas w/Flares"]
SA_COLORS = ['grey', 'k', vplot.colors.orange, vplot.colors.orange]
DA_ALPHAS = [1.0, 1.0, 0.5, 1.0]


def fnPlotHistograms(listData):
    """Plot step histograms for all model variants."""
    fig = plt.figure(figsize=(6.5, 6))
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


def main():
    """Generate cumulative XUV flux histogram for all model variants."""
    sOutputPath = (sys.argv[1] if len(sys.argv) > 1
                   else 'GJ1132b_CumulativeXUV_Multi.png')
    listData = [ftGatherFluxes(sDir) for sDir in SA_DIRECTORIES]
    fnPlotHistograms(listData)
    fnFormatAxes()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()
    fnPrintStatistics(listData)


if __name__ == "__main__":
    main()
