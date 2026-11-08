#!/usr/bin/env python3
"""
Summarize and plot GJ 1132's rotation-only age under several refit posteriors.

Each hyperparameter chain is pushed through EngleAgeDistribution/dataAge.py's
own age sampler (same P_rot distribution, scatter law, and 13 Gyr ceiling), so
the posteriors differ only in their hyperparameter draws. Reports the 2.5,
16, 50, 84 and 97.5 percentiles and overlays probability-density histograms.

Usage: python summarizeAgePosteriors.py --fit-summary <json>
       --chain <label>=<npy> [--chain ...] --output <json> --figure <path>
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

S_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(S_REPOSITORY, "EngleAgeDistribution"))
import dataAge  # noqa: E402

I_SEED = 42
DA_PERCENTILES = [2.5, 16, 50, 84, 97.5]
LIST_COLORS = [vplot.colors.dark_blue, vplot.colors.orange, vplot.colors.purple,
               vplot.colors.red]


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit-summary", required=True)
    parser.add_argument("--chain", required=True, action="append",
                        help="label=path to a 12-column hyperparameter chain (.npy)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure", required=True)
    return parser.parse_args()


def fdaComputeAgesGyr(daChain, dictSummary):
    """Push one chain through dataAge.py's sampler and return ages in Gyr."""
    np.random.seed(I_SEED)
    daLogAge = dataAge.fdaComputeRotationOnlyAges(daChain, dictSummary)
    return 10 ** daLogAge[daLogAge <= dataAge.D_MAX_LOG_AGE]


def fnPlotAges(dictAges, sFigurePath):
    """Overlay probability-density histograms of every age posterior."""
    daBins = np.arange(4.0, 11.0, 0.05)
    figure, axis = plt.subplots(figsize=(3.25, 3.0))
    for sColor, (sLabel, daAge) in zip(LIST_COLORS, dictAges.items()):
        daDensity, _ = np.histogram(daAge, bins=daBins, density=True)
        axis.stairs(daDensity, daBins, color=sColor, linewidth=1.3, label=sLabel)
    axis.set_xlabel("Age [Gyr]", fontsize=11)
    axis.set_ylabel(r"Probability density [Gyr$^{-1}$]", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=6.5, frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(sFigurePath, dpi=200)
    plt.close(figure)


def main():
    """Summarize every chain's age posterior and plot them together."""
    args = ftParseArguments()
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    dictAges = {}
    for sEntry in args.chain:
        sLabel, sPath = sEntry.split("=", 1)
        dictAges[sLabel] = fdaComputeAgesGyr(np.load(sPath), dictSummary)
    dictReport = {sLabel: {"daPercentiles": DA_PERCENTILES,
                           "daAgeGyr": np.percentile(daAge, DA_PERCENTILES).tolist()}
                  for sLabel, daAge in dictAges.items()}
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    fnPlotAges(dictAges, args.figure)
    print(f"Saved {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
