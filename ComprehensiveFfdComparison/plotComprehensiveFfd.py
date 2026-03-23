#!/usr/bin/env python
"""Plot comprehensive FFD comparison for GJ 1132 (Step 5).

Generates a single figure comparing the TESS-measured FFD of GJ 1132
with literature M-dwarf data, Ilin+2020 cluster observations, and the
Kepler-predicted FFD at 8 Gyr.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flareAnalysis import (
    fdFfdFit,
    fdFlareEquation,
    fdaInverseFfd,
    fdaComputeTessFitBand,
    fdictLoadKeplerPosterior,
    ftRunPipeline,
)

matplotlib = plt.matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})


# ---------------------------------------------------------------------------
# Overlay helpers (each under 20 lines)
# ---------------------------------------------------------------------------

def fnPlotTessFfdWithBand(ax, daFfdX, daFfdY, daFfdYerr,
                          dAlpha, dBeta, dictFitResults):
    """Draw TESS data points, best-fit line, and 1-sigma band."""
    iNumPoints = 100
    daLogEnergyData = np.linspace(np.min(daFfdX), np.max(daFfdX),
                                  iNumPoints)
    daLower, daUpper = fdaComputeTessFitBand(
        daLogEnergyData, dAlpha, dBeta, dictFitResults['pcov'])
    ax.fill_between(daLogEnergyData, daLower, daUpper,
                    color='k', alpha=0.15, label=r'TESS fit 1$\sigma$')
    ax.errorbar(daFfdX, daFfdY, yerr=daFfdYerr, linestyle='none', color='k')
    ax.scatter(daFfdX, daFfdY, c='k')
    ax.plot(daLogEnergyData, fdFfdFit(daLogEnergyData, dAlpha, dBeta), c='k')
    ax.text(31.1, -2.7, 'GJ 1132\n(observed)', color='k', fontsize=10)


def fnPlotLiteratureStars(ax, dictLiterature):
    """Overlay GJ 4083 and GJ 1243 data from Hawley+2014."""
    ax.plot(dictLiterature['gj4083_x'],
            np.log10(dictLiterature['gj4083_y']),
            marker='s', linestyle='dotted', lw=2,
            c=vplot.colors.pale_blue)
    ax.text(30.5, -2, 'GJ 4083',
            color=vplot.colors.pale_blue, fontsize=10)
    ax.plot(dictLiterature['gj1243_x'],
            np.log10(dictLiterature['gj1243_y']),
            c=vplot.colors.pale_blue, linestyle='dotted', lw=2)
    ax.text(31.5, 0.2, 'GJ 1243',
            color=vplot.colors.pale_blue, fontsize=10)


def fnPlotSingleCluster(ax, dictCluster, iIndex, daParams):
    """Plot observed and modeled FFD for one Ilin+2020 cluster."""
    k = iIndex
    daLogEnergyRange = np.log10(
        [dictCluster['Emin'][k], dictCluster['Emax'][k]])
    daObservedRate = np.log10(
        fdaInverseFfd([dictCluster['Emin'][k], dictCluster['Emax'][k]],
                      dictCluster['alpha'][k],
                      dictCluster['beta'][k])) - np.log10(365.25)
    ax.plot(daLogEnergyRange, daObservedRate,
            c=dictCluster['colors'][k],
            label=dictCluster['cluster'][k])
    tInput = (daLogEnergyRange,
              np.log10([dictCluster['ages'][k], dictCluster['ages'][k]]),
              np.array([0.2, 0.2]))
    ax.plot(daLogEnergyRange, fdFlareEquation(tInput, *daParams),
            c=dictCluster['colors'][k], linestyle='--')


def fnPlotKeplerPrediction(ax, daParams):
    """Overlay the Kepler-predicted FFD at 8 Gyr for a 0.2 Msun star."""
    iNumPoints = 100
    daLogEnergy = np.linspace(30.0, 34.5, iNumPoints)
    dLogAge = np.log10(8000)
    dMass = 0.2
    tInput = (daLogEnergy, np.full(iNumPoints, dLogAge),
              np.full(iNumPoints, dMass))
    ax.plot(daLogEnergy, fdFlareEquation(tInput, *daParams),
            color=vplot.colors.purple,
            label='Kepler prediction (8 Gyr)')
    ax.text(31.4, -1, 'GJ 1132\n(predicted)',
            color=vplot.colors.purple, fontsize=10)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def fnPlotComprehensiveFfd(daFfdX, daFfdY, daFfdYerr,
                           dAlpha, dBeta, dictLiterature,
                           dictCluster, dictFitResults,
                           sOutputPath=None):
    """Create the comprehensive FFD comparison figure."""
    daParams = dictCluster['params']
    fig, ax = plt.subplots()

    fnPlotTessFfdWithBand(ax, daFfdX, daFfdY, daFfdYerr,
                          dAlpha, dBeta, dictFitResults)
    fnPlotLiteratureStars(ax, dictLiterature)
    for iClusterIndex in [1, 0, 2]:
        fnPlotSingleCluster(ax, dictCluster, iClusterIndex, daParams)
    ax.plot([], c='k', linestyle='--', label='Fit at Cluster Ages')
    fnPlotKeplerPrediction(ax, daParams)

    ax.set_xlabel('log Flare Energy [erg]')
    ax.set_ylabel('log Flare Rate [day$^{-1}$]')
    ax.legend(fontsize=10)

    sFile = sOutputPath if sOutputPath else 'GJ1132_FFD_comp2.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()
    print(f"Saved comprehensive FFD comparison to {sFile}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fnParseArguments():
    """Parse command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Plot comprehensive FFD comparison for GJ 1132.")
    parser.add_argument(
        'sOutputPath',
        help="Output file path for the comparison figure.")
    parser.add_argument(
        '--flare-candidates', metavar='PATH',
        help="Path to flare_candidates.json.")
    parser.add_argument(
        '--kepler-posterior', metavar='PATH',
        help="Path to kepler_ffd_posterior_stats.json.")
    parser.add_argument(
        '--tess-cache-dir', metavar='PATH',
        help="Set lightkurve cache directory.")
    return parser.parse_args()


if __name__ == '__main__':
    args = fnParseArguments()

    if args.tess_cache_dir:
        os.environ['LIGHTKURVE_CACHE_DIR'] = args.tess_cache_dir

    dictKeplerPosterior = None
    if args.kepler_posterior:
        dictKeplerPosterior = fdictLoadKeplerPosterior(
            args.kepler_posterior)
        print(f"Loaded Kepler posterior from {args.kepler_posterior} "
              f"({dictKeplerPosterior['iNumSamples']} samples)")

    tPipeline = ftRunPipeline(
        dictKeplerPosterior=dictKeplerPosterior,
        sFlareJsonPath=args.flare_candidates)
    (_lc, _sectors, _tStart, _tStop, _sigma,
     daFfdX, daFfdY, _daFfdXerr, daFfdYerr,
     dictFitResults, dAlpha, dBeta,
     dictLiterature, dictCluster) = tPipeline

    fnPlotComprehensiveFfd(
        daFfdX, daFfdY, daFfdYerr, dAlpha, dBeta,
        dictLiterature, dictCluster, dictFitResults,
        sOutputPath=args.sOutputPath)
