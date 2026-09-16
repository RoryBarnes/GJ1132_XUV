#!/usr/bin/env python3
"""Plot MCMC posterior distributions from kepler_ffd.py output.

Reads saved samples and chain data, produces corner plot and trace
plots for the flare frequency distribution parameters.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import corner
import vplot


LIST_PARAM_NAMES = ["a1", "a2", "a3", "b1", "b2", "b3"]
LIST_PARAM_LABELS = [
    r"$a_1$", r"$a_2$", r"$a_3$",
    r"$b_1$", r"$b_2$", r"$b_3$",
]


def fnEnsureOutputDirectory(sOutputPath):
    """Create the parent directory of sOutputPath if needed."""
    sDirectory = os.path.dirname(sOutputPath)
    if sDirectory:
        os.makedirs(sDirectory, exist_ok=True)


def fnPlotCorner(daSamples, sOutputPath):
    """Create corner plot from MCMC samples."""
    fnEnsureOutputDirectory(sOutputPath)
    fig = corner.corner(
        daSamples,
        labels=LIST_PARAM_LABELS,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        label_kwargs={"fontsize": 20},
        labelpad=0.1,
    )
    iNumParams = daSamples.shape[1]
    daAxes = np.array(fig.axes).reshape((iNumParams, iNumParams))
    for i in range(iNumParams):
        for j in range(iNumParams):
            if i >= j:
                daAxes[i, j].tick_params(axis="both", labelsize=14)
    plt.savefig(sOutputPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved corner plot to {sOutputPath}")


def fnPlotTraces(daChain, listParamNames, sOutputPath):
    """Create trace plots from MCMC chain."""
    fnEnsureOutputDirectory(sOutputPath)
    iNumParams = len(listParamNames)
    iRows = (iNumParams + 2) // 3
    fig, daAxes = plt.subplots(iRows, 3, figsize=(15, 4 * iRows))
    daAxes = daAxes.flatten()
    for iParam in range(iNumParams):
        ax = daAxes[iParam]
        ax.plot(daChain[:, :, iParam], "k", alpha=0.3)
        ax.set_xlim(0, len(daChain))
        ax.set_ylabel(listParamNames[iParam])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    daAxes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trace plot to {sOutputPath}")


if __name__ == "__main__":
    print("Loading MCMC output for plotting...")

    sScriptDirectory = os.path.dirname(os.path.abspath(__file__))
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        sScriptDirectory, "Plot", "CornerVariableSlope.pdf"
    )

    sSamplesPath = os.path.join(
        sScriptDirectory, "flare_mcmc_samples.npy"
    )
    sChainPath = os.path.join(
        sScriptDirectory, "flare_mcmc_chain.npy"
    )

    daSamples = np.load(sSamplesPath)
    print(f"Loaded {daSamples.shape[0]} samples")

    fnPlotCorner(daSamples, sOutputPath)

    if os.path.exists(sChainPath):
        daChain = np.load(sChainPath)
        sTracePath = os.path.join(
            sScriptDirectory, "Plot",
            "flare_frequency_ensemble_traces.png",
        )
        fnPlotTraces(daChain, LIST_PARAM_NAMES, sTracePath)

    print("Plotting complete!")
