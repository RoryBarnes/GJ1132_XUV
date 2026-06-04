"""Plot the error source comparison figure."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import vplot

from utils.cumulativeXuv import (D_LOWER_BOUND, D_UPPER_BOUND,
                                  D_SHORELINE_FLUX, ftGatherFluxes)

S_OWN_OUTPUT_RELATIVE = "output/{stem}.json"
DICT_OWN_OUTPUT_STEMS = {
    "EngleModelErrorsOnly": "engle_model_errors_only_converged",
    "EngleStellarErrorsOnly": "engle_stellar_errors_only_converged",
    "RibasModelErrorsOnly": "ribas_model_errors_only_converged",
    "RibasStellarErrorsOnly": "ribas_stellar_errors_only_converged",
}


def flistPlotPanel(ax, listConvergedPaths, saLabels, saColors, sTitle):
    """Plot one panel of the error source comparison."""
    listData = [ftGatherFluxes(sPath) for sPath in listConvergedPaths]
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


def fsResolveOwnOutputPath(sScriptDirectory, sModelDirectory):
    """Return the absolute path to one of this step's own converged outputs."""
    sStem = DICT_OWN_OUTPUT_STEMS[sModelDirectory]
    sRelative = S_OWN_OUTPUT_RELATIVE.format(stem=sStem)
    return os.path.join(sScriptDirectory, sModelDirectory, sRelative)


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot error source comparison for cumulative XUV."
    )
    parser.add_argument("output_path",
                        help="Destination path for the figure.")
    parser.add_argument("--engle-converged", required=True,
                        help="Path to A10 Engle converged JSON.")
    parser.add_argument("--ribas-converged", required=True,
                        help="Path to A10 Ribas converged JSON.")
    return parser.parse_args()


def flistEngleVariantPaths(sScriptDirectory, args):
    """Return the three converged-JSON paths for the Engle panel."""
    return [
        args.engle_converged,
        fsResolveOwnOutputPath(sScriptDirectory, "EngleModelErrorsOnly"),
        fsResolveOwnOutputPath(sScriptDirectory, "EngleStellarErrorsOnly"),
    ]


def flistRibasVariantPaths(sScriptDirectory, args):
    """Return the three converged-JSON paths for the Ribas panel."""
    return [
        args.ribas_converged,
        fsResolveOwnOutputPath(sScriptDirectory, "RibasModelErrorsOnly"),
        fsResolveOwnOutputPath(sScriptDirectory, "RibasStellarErrorsOnly"),
    ]


def main():
    """Generate error source comparison figure."""
    args = ftParseArguments()
    sScriptDirectory = os.path.dirname(os.path.abspath(__file__))
    saLabels = ['All Errors', 'Model Errors Only', 'Stellar Errors Only']
    saColors = ['k', vplot.colors.purple, vplot.colors.red]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    listEnglePaths = flistEngleVariantPaths(sScriptDirectory, args)
    listEngle = flistPlotPanel(ax1, listEnglePaths, saLabels, saColors,
                               'Engle (2024)')
    ax1.set_ylabel('Fraction', fontsize=20)

    listRibasPaths = flistRibasVariantPaths(sScriptDirectory, args)
    listRibas = flistPlotPanel(ax2, listRibasPaths, saLabels, saColors,
                               'Ribas et al. (2005)')

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300)
    plt.close()

    fnPrintVariantStatistics("ENGLE MODEL VARIANTS", saLabels, listEngle)
    print()
    fnPrintVariantStatistics("RIBAS MODEL VARIANTS", saLabels, listRibas)


if __name__ == "__main__":
    main()
