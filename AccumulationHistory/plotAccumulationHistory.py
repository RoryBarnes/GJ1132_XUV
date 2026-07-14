#!/usr/bin/env python3
"""
Plot when GJ 1132 b climbed its cumulative-XUV hill.

Cumulative XUV flux versus stellar age. Thin lines are 25 real vconverge
trajectories, chosen at evenly spaced final-flux quantiles so they span the
distribution; each ends at its trial's own sampled present age, so the endpoint
dots are draws from the joint (present age, total dose) posterior. The heavier
line is the median accumulation history over the well-covered age range. The
age posterior's 95% interval is marked by dotted verticals: by the time the
star reaches even the lower edge, every trajectory has long since flattened, so
the planet received essentially its whole lifetime XUV dose billions of years
earlier. The cosmic shoreline is drawn horizontally to show the planet crossed
the atmosphere-stripping threshold early too.

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


def fnPlotShownTrajectories(axis, listShown):
    """Draw the 25 real trajectories with an endpoint dot at each present age."""
    for iIndex, dictTrajectory in enumerate(listShown):
        axis.plot(dictTrajectory["daAgeGrid"], dictTrajectory["daFlux"],
                  color=S_COLOR_HISTORY, linewidth=0.8, alpha=0.45,
                  label="25 sampled simulations" if iIndex == 0 else None)
    daEndAge = [d["dPresentAge"] for d in listShown]
    daEndFlux = [d["dFinalFlux"] for d in listShown]
    axis.scatter(daEndAge, daEndFlux, s=14, color=S_COLOR_HISTORY, zorder=5,
                 label="present-day state (age, dose)")


def fnPlotHistory(axis, dictHistory):
    """Draw the median accumulation curve over the well-covered age range."""
    axis.plot(dictHistory["daAgeGrid"], dictHistory["daMedianFlux"],
              color=S_COLOR_HISTORY, linewidth=2.4,
              label="median accumulation history")


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
    fnPlotShownTrajectories(axis, dictSummary["listShownTrajectories"])
    fnPlotHistory(axis, dictSummary["dictHistory"])
    fnPlotAgeInterval(axis, dictSummary["dictAgePosterior"],
                      dictSummary["dFractionByAgeLowerBound"])
    axis.axhline(D_SHORELINE_FLUX, color=S_COLOR_SHORELINE, linestyle="--",
                 linewidth=1.5, label="cosmic shoreline")
    dMaxFlux = max(d["dFinalFlux"]
                   for d in dictSummary["listShownTrajectories"])
    axis.set_yscale("log")
    axis.set_ylim(D_SHORELINE_FLUX * 0.3, dMaxFlux * 1.3)
    axis.set_xlim(0, max(d["dPresentAge"]
                         for d in dictSummary["listShownTrajectories"]) * 1.02)
    axis.set_xlabel("stellar age [Gyr]")
    axis.set_ylabel(r"cumulative XUV flux [$F_{\rm XUV,\oplus}$]")
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(sOutputPath, bbox_inches="tight")
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
