#!/usr/bin/env python3
"""Render the generative probabilistic graphical model (PGM) for GJ 1132.

Node conventions follow the RevBayes graphical-model tutorial:
  observed / clamped  -> shaded circle
  stochastic (latent) -> solid open circle
  deterministic       -> dashed circle
  constant / hyper    -> square
Plates denote replication (calibrator stars, Kepler M dwarfs, posterior draws).

The canvas is sized purely for legibility (not to an article page): node
coordinates are spread generously so no node, label, or plate overlaps. The
node/edge/plate structure and the parameter categorization are loaded from the
JSON emitted by ``dataModelGraph.py``.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import vplot
from daft import PGM

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
D_SHADED = "0.72"
F_PAD = 7.0
F_GRID_UNIT = 1.9
F_NODE_FONT = 20


def fdictLoadStructure():
    """Load the PGM structure JSON produced by the data step."""
    sPath = os.path.join(S_DIRECTORY,
                         "probabilisticGraphicalModelStructure.json")
    with open(sPath) as fileHandle:
        return json.load(fileHandle)


def fdictNodeKwargs(sCategory):
    """Map a parameter category to daft add_node keyword arguments."""
    if sCategory == "observed":
        return {"observed": True, "shape": "ellipse"}
    if sCategory == "deterministic":
        return {"plot_params": {"linestyle": "dashed", "linewidth": 1.6},
                "shape": "ellipse"}
    if sCategory == "constant":
        return {"shape": "rectangle",
                "plot_params": {"linewidth": 1.3, "facecolor": "#f0f0f0"}}
    return {"shape": "ellipse", "plot_params": {"linewidth": 1.6}}


def fnAddNodes(oPgm, listNodes):
    """Add every node to the PGM with its category-specific styling."""
    for dictNode in listNodes:
        dScale = 3.38 if dictNode["sCategory"] == "constant" else 3.05
        oPgm.add_node(dictNode["sId"], dictNode["sLabel"], dictNode["fX"],
                      dictNode["fY"], scale=dScale, fontsize=F_NODE_FONT,
                      **fdictNodeKwargs(dictNode["sCategory"]))


def fnAddEdges(oPgm, listEdges):
    """Add every directed conditional-dependence edge."""
    for dictEdge in listEdges:
        oPgm.add_edge(dictEdge["sSrc"], dictEdge["sDst"], linewidth=1.1)


def fnAddPlates(oPgm, listPlates):
    """Add every replication plate rectangle with its count label."""
    for dictPlate in listPlates:
        oPgm.add_plate(
            [dictPlate["fX"], dictPlate["fY"], dictPlate["fWidth"],
             dictPlate["fHeight"]], label=dictPlate["sLabel"], fontsize=20,
            position=dictPlate.get("sPosition", "bottom right"))


def ftComputeExtent(listNodes, listPlates):
    """Return (minX, maxX, minY, maxY) spanning all nodes and plates."""
    listX = [dictNode["fX"] for dictNode in listNodes]
    listY = [dictNode["fY"] for dictNode in listNodes]
    for dictPlate in listPlates:
        listX += [dictPlate["fX"], dictPlate["fX"] + dictPlate["fWidth"]]
        listY += [dictPlate["fY"], dictPlate["fY"] + dictPlate["fHeight"]]
    return min(listX), max(listX), min(listY), max(listY)


def ftCanvas(listNodes, listPlates):
    """Return (shape, origin) padded to leave clear corners for annotations."""
    fMinX, fMaxX, fMinY, fMaxY = ftComputeExtent(listNodes, listPlates)
    tOrigin = [fMinX - F_PAD, fMinY - F_PAD]
    tShape = [(fMaxX - fMinX) + 2 * F_PAD, (fMaxY - fMinY) + 2 * F_PAD]
    return tShape, tOrigin


def fnAddLegend(axis):
    """Draw a manual legend keyed to the RevBayes node conventions."""
    listHandles = [
        Circle((0, 0), 0.1, facecolor=D_SHADED, edgecolor="k",
               label="observed / clamped (data)"),
        Circle((0, 0), 0.1, facecolor="white", edgecolor="k",
               label="stochastic / latent (inferred)"),
        Circle((0, 0), 0.1, facecolor="white", edgecolor="k", linestyle="--",
               label="deterministic (function of parents)"),
        Patch(facecolor="#f0f0f0", edgecolor="k",
              label="constant / hyperparameter"),
    ]
    axis.legend(handles=listHandles, loc="upper left", fontsize=20,
                frameon=True, handlelength=1.4, borderpad=0.7,
                bbox_to_anchor=(0.008, 0.992),
                title="Node type (RevBayes convention)", title_fontsize=21)


def fnAddGlossary(axis):
    """Add a compact glossary mapping the short symbols to their meaning."""
    sText = (
        r"$\theta_{rot},\theta_{xuv}$: Engle EMD coeffs (+ intrinsic scatter)"
        "\n" r"$\theta_{FFD}$: Davenport FFD coeffs $a_1..b_3$"
        "\n" r"$\tau$: log age;   $\alpha_k$: per-star latent age"
        "\n" r"$q_{\rm E},q_{\rm R}$: quiescent $L_{XUV}/L_{bol}$ (EMD, Ribas)"
        "\n" r"$L^{fl}_{XUV},L^{tot}_{XUV}$: flare, total XUV luminosity"
        "\n" r"$F_{XUV}(t)$: flux at planet;   $F^{cum}_{XUV}$: cumulative flux")
    axis.text(0.992, 0.008, sText, transform=axis.transAxes, fontsize=18,
              va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="#999999", alpha=0.95))


def fnRenderPgm(sOutputPath):
    """Assemble the full PGM figure and save it as a vector PDF."""
    dictStructure = fdictLoadStructure()
    tShape, tOrigin = ftCanvas(dictStructure["listNodes"],
                               dictStructure["listPlates"])
    oPgm = PGM(shape=tShape, origin=tOrigin, grid_unit=F_GRID_UNIT,
               observed_style="shaded", node_ec="k", directed=True)
    fnAddNodes(oPgm, dictStructure["listNodes"])
    fnAddEdges(oPgm, dictStructure["listEdges"])
    fnAddPlates(oPgm, dictStructure["listPlates"])
    oPgm.render()
    fnAddLegend(oPgm.ax)
    fnAddGlossary(oPgm.ax)
    os.makedirs(os.path.dirname(os.path.abspath(sOutputPath)), exist_ok=True)
    oPgm.savefig(sOutputPath, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {sOutputPath}")


if __name__ == "__main__":
    sOut = sys.argv[1] if len(sys.argv) > 1 else "ProbabilisticGraphicalModel.pdf"
    fnRenderPgm(sOut)
