import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import vplot

from utils.cumulativeXuv import (D_LOWER_BOUND, D_UPPER_BOUND,
                                  D_SHORELINE_FLUX, ftGatherFluxes)


def flistPlotPanel(ax, saDirectories, saLabels, saColors, sTitle):
    """Plot one panel of the error source comparison."""
    listData = [ftGatherFluxes(sDir) for sDir in saDirectories]
    ax.axvline(D_SHORELINE_FLUX, color=vplot.colors.pale_blue, linewidth=6)
    for i, (daBins, daFractions, _, _, _) in enumerate(listData):
        ax.step(daBins, daFractions, where='mid', color=saColors[i],
                linestyle='-', linewidth=2, label=saLabels[i])
    fnFormatPanel(ax, sTitle)
    return listData


def fnFormatPanel(ax, sTitle):
    """Apply axis formatting to a single panel."""
    ax.set_xlabel('Normalized Cumulative XUV Flux', fontsize=20)
    ax.set_xlim(D_LOWER_BOUND, D_UPPER_BOUND)
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylim(0, 0.4)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(sTitle, fontsize=22)
    ax.annotate('Cosmic Shoreline', (40, 0.06), fontsize=20,
                rotation=90, color=vplot.colors.pale_blue)


def fnPrintVariantStatistics(sGroupName, saLabels, listData):
    """Print mean and confidence interval for a group of variants."""
    print(f"{sGroupName}:")
    for i, sLabel in enumerate(saLabels):
        dMean, dLower, dUpper = listData[i][2], listData[i][3], listData[i][4]
        print(f"  {sLabel:25s} - Mean: {dMean:.2f}, "
              f"95% CI: [{dLower:.2f}, {dUpper:.2f}]")


def main():
    """Generate error source comparison figure."""
    sOutputPath = (sys.argv[1] if len(sys.argv) > 1
                   else 'GJ1132b_ErrorSourceComparison.pdf')
    saLabels = ['All Errors', 'Model Errors Only', 'Stellar Errors Only']
    saColors = ['k', vplot.colors.purple, vplot.colors.red]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    saEngleDirs = ['Engle', 'EngleModelErrorsOnly', 'EngleStellarErrorsOnly']
    listEngle = flistPlotPanel(ax1, saEngleDirs, saLabels, saColors, 'Engle (2024)')
    ax1.set_ylabel('Fraction', fontsize=20)

    saRibasDirs = ['Ribas', 'RibasModelErrorsOnly', 'RibasStellarErrorsOnly']
    listRibas = flistPlotPanel(ax2, saRibasDirs, saLabels, saColors, 'Ribas et al. (2005)')

    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()

    fnPrintVariantStatistics("ENGLE MODEL VARIANTS", saLabels, listEngle)
    print()
    fnPrintVariantStatistics("RIBAS MODEL VARIANTS", saLabels, listRibas)


if __name__ == "__main__":
    main()
