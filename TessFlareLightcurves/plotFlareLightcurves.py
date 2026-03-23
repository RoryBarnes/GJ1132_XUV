#!/usr/bin/env python
"""Plot TESS flare lightcurves for GJ 1132 (Step 4).

Generates a multi-panel figure showing each detected flare in the
normalized TESS lightcurve with the flare cadences highlighted.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flareAnalysis import ftRunPipeline

matplotlib = plt.matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def ftComputePanelLayout(iNumFlares):
    """Return (iRows, iCols) for the flare lightcurve grid."""
    if iNumFlares <= 3:
        return 1, iNumFlares
    iCols = (iNumFlares + 1) // 2
    return 2, iCols


# ---------------------------------------------------------------------------
# Per-panel plotting
# ---------------------------------------------------------------------------

def fnPlotSingleFlarePanel(ax, lightcurveCollection, iSector,
                           dTimeStart, dTimeStop, iCol, iCols,
                           dPeakSigma=None):
    """Draw one flare panel on the given axes."""
    daFlareIdx = np.where(
        (lightcurveCollection[iSector]['time'].value >= dTimeStart)
        & (lightcurveCollection[iSector]['time'].value <= dTimeStop)
    )[0]
    daTime = lightcurveCollection[iSector]['time'].value
    daFlux = lightcurveCollection[iSector].normalize()['flux'].value
    dTimeOffset = np.nanmin(daTime[daFlareIdx])

    ax.plot(daTime - dTimeOffset, daFlux, c='k')
    ax.scatter(daTime[daFlareIdx] - dTimeOffset,
               daFlux[daFlareIdx], c=vplot.colors.pale_blue)
    ax.set_xlim(-0.05, 0.1)
    ax.set_xlabel('Time [days]', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    if iCol == 0:
        ax.set_ylabel('Relative Flux', fontsize=20)
    if dPeakSigma is not None:
        ax.text(0.05, 0.95, f'{dPeakSigma:.1f}\u03c3',
                transform=ax.transAxes, fontsize=16,
                verticalalignment='top')


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def fnPlotFlareLightcurves(lightcurveCollection, listSectors,
                           daTimeStart, daTimeStop,
                           daPeakSigma=None, sOutputPath=None):
    """Create multi-panel flare lightcurve figure."""
    iNumFlares = len(daTimeStart)
    iRows, iCols = ftComputePanelLayout(iNumFlares)
    fig, axes = plt.subplots(
        iRows, iCols,
        figsize=(5 * iCols, 5 * iRows), sharey=True)
    daAxesFlat = np.atleast_1d(axes).flatten()

    for k in range(iNumFlares):
        dSigma = daPeakSigma[k] if daPeakSigma is not None else None
        fnPlotSingleFlarePanel(
            daAxesFlat[k], lightcurveCollection, listSectors[k],
            daTimeStart[k], daTimeStop[k], k % iCols, iCols,
            dPeakSigma=dSigma)

    for k in range(iNumFlares, iRows * iCols):
        daAxesFlat[k].set_visible(False)

    plt.tight_layout()
    sFile = sOutputPath if sOutputPath else 'GJ1132_flares.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()
    print(f"Saved flare lightcurves to {sFile}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fnParseArguments():
    """Parse command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Plot TESS flare lightcurves for GJ 1132.")
    parser.add_argument(
        'sOutputPath',
        help="Output file path for the lightcurve figure.")
    parser.add_argument(
        '--flare-candidates', metavar='PATH',
        help="Path to flare_candidates.json.")
    parser.add_argument(
        '--tess-cache-dir', metavar='PATH',
        help="Set lightkurve cache directory.")
    return parser.parse_args()


if __name__ == '__main__':
    args = fnParseArguments()

    if args.tess_cache_dir:
        os.environ['LIGHTKURVE_CACHE_DIR'] = args.tess_cache_dir

    tPipeline = ftRunPipeline(sFlareJsonPath=args.flare_candidates)
    (lightcurveCollection, listSectors, daTimeStart, daTimeStop,
     daPeakSigma, _daFfdX, _daFfdY, _daFfdXerr, _daFfdYerr,
     _dictFit, _dAlpha, _dBeta, _dictLit, _dictCluster) = tPipeline

    fnPlotFlareLightcurves(
        lightcurveCollection, listSectors, daTimeStart, daTimeStop,
        daPeakSigma=daPeakSigma, sOutputPath=args.sOutputPath)
