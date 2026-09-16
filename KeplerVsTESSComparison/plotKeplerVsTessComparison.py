#!/usr/bin/env python
"""Plot Kepler vs TESS alpha-beta parameter comparison (Step 13).

Generates a scatter plot of (alpha, beta) draws from the
Kepler-predicted and TESS-measured FFD posteriors for GJ 1132,
and prints the joint tension statistic.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import vplot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flareAnalysis import (
    fdictLoadKeplerPosterior,
    fnPrintKeplerTessDiscrepancy,
    ftRunPipeline,
)

matplotlib = plt.matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})


# ---------------------------------------------------------------------------
# Default Kepler parameters (when no posterior JSON is available)
# ---------------------------------------------------------------------------

def fdaGetDefaultKeplerMeans():
    """Return hard-coded Kepler means for the 6-parameter model."""
    return np.array([-0.148, 0.517, -0.618, 4.69, -16.45, 19.446])


def fdaGetDefaultKeplerCovariance():
    """Return a diagonal covariance built from hard-coded uncertainties."""
    daUncertainties = np.array(
        [0.000529, 0.00244, 0.00231, 0.0180, 0.0824, 0.0779])
    return np.diag(daUncertainties ** 2)


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def ftDrawKeplerAlphaBeta(daKeplerMeans, daKeplerCovariance,
                          iNumDraws=100):
    """Draw (alpha, beta) samples from the Kepler posterior at 8 Gyr."""
    dLogAge = np.log10(8000)
    dMass = 0.2
    daSamples = np.random.multivariate_normal(
        daKeplerMeans, daKeplerCovariance, size=iNumDraws)
    daAlphas = (daSamples[:, 0] * dLogAge
                + daSamples[:, 1] * dMass
                + daSamples[:, 2])
    daBetas = (daSamples[:, 3] * dLogAge
               + daSamples[:, 4] * dMass
               + daSamples[:, 5])
    return daAlphas, daBetas


def ftDrawTessAlphaBeta(dAlpha, dBeta, daCovarianceMatrix,
                        iNumDraws=100):
    """Draw (alpha, beta) samples from the TESS FFD fit."""
    daSamples = np.random.multivariate_normal(
        [dAlpha, dBeta], daCovarianceMatrix, size=iNumDraws)
    return daSamples[:, 0], daSamples[:, 1]


# ---------------------------------------------------------------------------
# Plot function
# ---------------------------------------------------------------------------

def fnPlotAlphaBetaScatter(daKeplerAlphas, daKeplerBetas,
                           daTessAlphas, daTessBetas,
                           sOutputPath=None):
    """Create the alpha-beta scatter comparison figure."""
    plt.figure(figsize=(10, 8))
    plt.scatter(daKeplerAlphas, daKeplerBetas,
                c=vplot.colors.orange, alpha=0.6, s=50,
                label=r'$Kepler$ (8 Gyr)',
                edgecolors='k', linewidths=0.5)
    plt.scatter(daTessAlphas, daTessBetas,
                c=vplot.colors.pale_blue, alpha=0.6, s=50,
                label=r'GJ 1132 ($TESS$)',
                edgecolors='k', linewidths=0.5)
    plt.scatter([-0.32], [8.54], label='Best fit',
                marker='^', color='red', s=100)
    plt.xlabel(r'Slope ($a$)', fontsize=18)
    plt.ylabel(r'Intercept ($b$)', fontsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.tick_params(axis='both', labelsize=14)

    sFile = sOutputPath if sOutputPath else 'FitComparison.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()
    print(f"Saved alpha-beta comparison to {sFile}")


def fnPrintSummaryStatistics(dAlpha, dBeta, dictFitResults,
                             daKeplerAlphas, daKeplerBetas):
    """Print Kepler vs TESS summary statistics."""
    print(f"Kepler prediction (8 Gyr): alpha = "
          f"{np.mean(daKeplerAlphas):.4f} +/- "
          f"{np.std(daKeplerAlphas):.4f}, "
          f"beta = {np.mean(daKeplerBetas):.4f} +/- "
          f"{np.std(daKeplerBetas):.4f}")
    print(f"GJ 1132 measurement: alpha = {dAlpha:.4f} +/- "
          f"{dictFitResults['alpha_err']:.4f}, "
          f"beta = {dBeta:.4f} +/- "
          f"{dictFitResults['beta_err']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fnParseArguments():
    """Parse command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Plot Kepler vs TESS alpha-beta comparison.")
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
     _daFfdX, _daFfdY, _daFfdXerr, _daFfdYerr,
     dictFitResults, dAlpha, dBeta,
     _dictLit, _dictCluster) = tPipeline

    if dictKeplerPosterior is not None:
        daKeplerMeans = dictKeplerPosterior["daMedians"]
        daKeplerCovariance = dictKeplerPosterior["daCovarianceMatrix"]
    else:
        daKeplerMeans = fdaGetDefaultKeplerMeans()
        daKeplerCovariance = fdaGetDefaultKeplerCovariance()

    daKeplerAlphas, daKeplerBetas = ftDrawKeplerAlphaBeta(
        daKeplerMeans, daKeplerCovariance)
    daTessAlphas, daTessBetas = ftDrawTessAlphaBeta(
        dAlpha, dBeta, dictFitResults['pcov'])

    fnPlotAlphaBetaScatter(
        daKeplerAlphas, daKeplerBetas,
        daTessAlphas, daTessBetas,
        sOutputPath=args.sOutputPath)

    fnPrintSummaryStatistics(
        dAlpha, dBeta, dictFitResults,
        daKeplerAlphas, daKeplerBetas)

    if dictKeplerPosterior is not None:
        fnPrintKeplerTessDiscrepancy(
            dictFitResults, daKeplerMeans, daKeplerCovariance)
