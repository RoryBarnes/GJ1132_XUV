#!/usr/bin/env python3
"""
Plot when GJ 1132 b climbed its cumulative-XUV hill.

Cumulative XUV flux versus stellar age: the median accumulation history with
its 68% envelope, reconstructed from the Engle sweep's forward trajectories.
The age posterior's 95% interval is marked by dotted vertical lines; by the
time the star reaches even the lower edge of that interval the curve has long
since flattened, so the planet received essentially its entire lifetime XUV
dose billions of years earlier. The cosmic shoreline is drawn as a horizontal
line to show the planet crossed the atmosphere-stripping threshold early too.

Usage: python plotAccumulationHistory.py <outputPath>
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import D_SHORELINE_FLUX

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

S_COLOR_HISTORY = vplot.colors.dark_blue
S_COLOR_AGE = vplot.colors.orange
S_COLOR_SHORELINE = vplot.colors.red


def fdictLoadHistory():
    """Load the accumulation-history product."""
    with open(os.path.join(S_DIRECTORY,
                           "accumulationHistory.json")) as fileHandle:
        return json.load(fileHandle)


def fnPlotHistory(axis, dictHistory):
    """Draw the median accumulation curve and its 68% envelope."""
    daAge = np.array(dictHistory["daAgeGrid"])
    axis.fill_between(daAge, dictHistory["daLowerFlux"],
                      dictHistory["daUpperFlux"], color=S_COLOR_HISTORY,
                      alpha=0.22, linewidth=0, label="68% of realizations")
    axis.plot(daAge, dictHistory["daMedianFlux"], color=S_COLOR_HISTORY,
              linewidth=2.2, label="median accumulation history")


def fnPlotAgeInterval(axis, dictAge, dFractionLower):
    """Mark the age posterior's 95% interval with dotted verticals."""
    for dEdge in dictAge["ci95"]:
        axis.axvline(dEdge, color=S_COLOR_AGE, linestyle=":", linewidth=1.8)
    axis.axvspan(dictAge["ci95"][0], dictAge["ci95"][1], color=S_COLOR_AGE,
                 alpha=0.08, linewidth=0)
    dMid = np.sqrt(dictAge["ci95"][0] * dictAge["ci95"][1])
    axis.text(dMid, 0.05, "age\n95% CI", color=S_COLOR_AGE, ha="center",
              va="bottom", fontsize=8, transform=axis.get_xaxis_transform())
    axis.annotate(f"{100 * dFractionLower:.0f}% of the dose\n"
                  "already delivered",
                  xy=(dictAge["ci95"][0], 0.62), xytext=(0.5, 0.35),
                  textcoords=axis.transAxes, xycoords=axis.get_xaxis_transform(),
                  ha="center", fontsize=8.5, color="0.25",
                  arrowprops=dict(arrowstyle="->", color="0.5", linewidth=1))


def main():
    """Render the accumulation-history figure."""
    sOutputPath = sys.argv[1]
    dictSummary = fdictLoadHistory()
    os.makedirs(os.path.dirname(sOutputPath), exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.6))
    fnPlotHistory(axis, dictSummary["dictHistory"])
    fnPlotAgeInterval(axis, dictSummary["dictAgePosterior"],
                      dictSummary["dFractionByAgeLowerBound"])
    axis.axhline(D_SHORELINE_FLUX, color=S_COLOR_SHORELINE, linestyle="--",
                 linewidth=1.5, label="cosmic shoreline")
    axis.set_ylim(0, max(dictSummary["dictHistory"]["daUpperFlux"]) * 1.05)
    axis.set_xlim(0, dictSummary["dictAgePosterior"]["ci95"][1] * 1.02)
    axis.set_xlabel("stellar age [Gyr]")
    axis.set_ylabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
