#!/usr/bin/env python3
"""
Plot XUV luminosity evolution for 100 random samples from Engle and Ribas models.

Randomly selects 100 vplanet outputs from each of the EngleBarnes and RibasBarnes
directories and creates a two-panel plot comparing XUV luminosity evolution over time.
"""

import glob
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vplot

I_LABEL_FONT = 20
I_TICK_FONT = 14
I_NUM_SAMPLES = 100
S_FORWARD_FILE = "gj1132.star.forward"


def ftLoadXuvData(sFilepath):
    """Load time and total XUV luminosity from a vplanet forward file."""
    try:
        daData = np.loadtxt(sFilepath)
        daTime = daData[:, 0]
        daXuvTotal = daData[:, 4]
        return daTime, daXuvTotal
    except Exception as error:
        print(f"Error loading {sFilepath}: {error}")
        return None, None


def flistGetOutputSubdirectories(sOutputDirectory):
    """Return all subdirectory paths under *sOutputDirectory*."""
    return [sPath for sPath in glob.glob(os.path.join(sOutputDirectory, '*'))
            if os.path.isdir(sPath)]


def fiPlotEvolutionCurves(ax, listSubdirectories):
    """Plot XUV evolution curves on *ax*, returning the count of plotted runs."""
    iCount = 0
    for sSubdir in listSubdirectories:
        sFilepath = os.path.join(sSubdir, S_FORWARD_FILE)
        if not os.path.exists(sFilepath):
            continue
        daTime, daXuvTotal = ftLoadXuvData(sFilepath)
        if daTime is not None and daXuvTotal is not None:
            ax.plot(daTime, daXuvTotal, color='k', alpha=0.15, linewidth=0.8)
            iCount += 1
    return iCount


def fnFormatPanel(ax, sTitle, bShowYlabel=False):
    """Apply log-log formatting and labels to a single panel."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time [years]', fontsize=I_LABEL_FONT)
    if bShowYlabel:
        ax.set_ylabel(r'Total XUV Luminosity [$L_\odot$]', fontsize=I_LABEL_FONT)
    ax.set_title(sTitle, fontsize=I_LABEL_FONT)
    ax.tick_params(axis='both', labelsize=I_TICK_FONT)


def fnMatchYlimits(ax1, ax2):
    """Set matching y-axis limits across two panels."""
    dYmin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    dYmax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(dYmin, dYmax)
    ax2.set_ylim(dYmin, dYmax)


def flistSampleSubdirectories(sModelDirectory):
    """Get subdirectories and randomly sample up to I_NUM_SAMPLES."""
    sOutputDirectory = str(Path(__file__).parent / sModelDirectory / "output")
    listSubdirs = flistGetOutputSubdirectories(sOutputDirectory)
    print(f"Found {len(listSubdirs)} {sModelDirectory} directories")
    if len(listSubdirs) == 0:
        raise FileNotFoundError(
            f"No output subdirectories found in {sOutputDirectory}")
    return random.sample(listSubdirs, min(I_NUM_SAMPLES, len(listSubdirs)))


def main(sOutputPath=None):
    """Generate the XUV evolution comparison figure."""
    random.seed(42)
    np.random.seed(42)

    listEngleSample = flistSampleSubdirectories("EngleBarnes")
    listRibasSample = flistSampleSubdirectories("RibasBarnes")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    iEngleCount = fiPlotEvolutionCurves(ax1, listEngleSample)
    iRibasCount = fiPlotEvolutionCurves(ax2, listRibasSample)
    print(f"Plotted {iEngleCount} EngleBarnes, {iRibasCount} RibasBarnes runs")

    fnFormatPanel(ax1, 'Engle (2024)', bShowYlabel=True)
    fnFormatPanel(ax2, 'Ribas et al. (2005)')
    fnMatchYlimits(ax1, ax2)

    plt.tight_layout()
    sPath = sOutputPath if sOutputPath else str(Path(__file__).parent / "XUVEvol.pdf")
    plt.savefig(sPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {sPath}")


if __name__ == "__main__":
    sOutputPath = sys.argv[1] if len(sys.argv) > 1 else None
    main(sOutputPath=sOutputPath)
