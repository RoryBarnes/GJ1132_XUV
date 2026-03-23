#!/usr/bin/env python3
"""
Plot XUV luminosity distribution histograms from pre-computed sample files.

Reads lxuv_samples.txt, lxuv_lbol_samples.txt, and log_lxuv_lbol_samples.txt,
then creates normalized histograms with Gaussian overlays.
"""

import argparse
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import vplot

D_FIG_SIZE_X = 3.25
D_FIG_SIZE_Y = 3
I_NUM_BINS = 50


def ftComputeStatistics(daSamples):
    """Return (dMean, dStd, daCI) for a sample array."""
    dMean = np.mean(daSamples)
    dStd = np.std(daSamples)
    daCI = np.percentile(daSamples, [2.5, 97.5])
    return dMean, dStd, daCI


def fnPlotStepHistogram(daSamples, dMean, dStd, dXscale=1.0):
    """Plot a step histogram with Gaussian overlay on the current figure."""
    daScaled = daSamples * dXscale
    dScaledMean = dMean * dXscale
    dScaledStd = dStd * dXscale
    daCounts, daBinEdges = np.histogram(daScaled, bins=I_NUM_BINS)
    daFractions = daCounts / len(daScaled)
    plt.step(daBinEdges[:-1], daFractions, where='mid',
             color='k', linewidth=1.5, label='Data')
    daXgauss = np.linspace(daBinEdges[0], daBinEdges[-1], 200)
    dBinWidth = daBinEdges[1] - daBinEdges[0]
    daGaussPdf = stats.norm.pdf(daXgauss, dScaledMean, dScaledStd) * dBinWidth
    plt.plot(daXgauss, daGaussPdf, color=vplot.colors.red,
             linestyle='dashed', linewidth=1.5, label="Fit")


def fnFormatNormalizedHistogram(sXlabel):
    """Apply standard formatting to a normalized histogram."""
    plt.xlabel(sXlabel, fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()


def fnPlotNormalizedHistogram(daSamples, dMean, dStd, sXlabel, sFilename,
                              dXscale=1.0):
    """Create and save a normalized histogram with Gaussian overlay."""
    plt.figure(figsize=(D_FIG_SIZE_X, D_FIG_SIZE_Y))
    fnPlotStepHistogram(daSamples, dMean, dStd, dXscale)
    fnFormatNormalizedHistogram(sXlabel)
    plt.savefig(sFilename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to: {sFilename}")


def fsResolvePath(sFilename, sOutputDirectory=None):
    """Resolve an output filename with an optional output directory."""
    if sOutputDirectory:
        return os.path.join(sOutputDirectory, sFilename)
    return sFilename


def main(sOutputDirectory=None, sFigureType="pdf"):
    """Load sample files and generate all L_XUV histogram figures."""
    daLxuvLsun = np.loadtxt('lxuv_samples.txt')
    dMean, dStd, _ = ftComputeStatistics(daLxuvLsun)
    fnPlotNormalizedHistogram(
        daLxuvLsun, dMean, dStd,
        sXlabel=r'$L_{XUV}$ [$10^{-7} L_\odot$]',
        sFilename=fsResolvePath(f'lxuv_hist.{sFigureType}', sOutputDirectory),
        dXscale=1e7)

    daRatio = np.loadtxt('lxuv_lbol_samples.txt')
    dRatioMean, dRatioStd, _ = ftComputeStatistics(daRatio)
    fnPlotNormalizedHistogram(
        daRatio, dRatioMean, dRatioStd,
        sXlabel=r'$L_{XUV} / L_{bol}$',
        sFilename=fsResolvePath(
            f'lxuv_lbol_hist.{sFigureType}', sOutputDirectory))

    daLogRatio = np.loadtxt('log_lxuv_lbol_samples.txt')
    dLogMean, dLogStd, _ = ftComputeStatistics(daLogRatio)
    fnPlotNormalizedHistogram(
        daLogRatio, dLogMean, dLogStd,
        sXlabel=r'$\log_{10}(L_{XUV} / L_{bol})$',
        sFilename=fsResolvePath(
            f'log_lxuv_lbol_hist.{sFigureType}', sOutputDirectory))

    print("\nPlotting complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot GJ 1132 XUV luminosity distribution histograms.")
    parser.add_argument('--output-directory', metavar='PATH',
                        help="Directory for output figure files.")
    parser.add_argument('--figure-type', default='pdf',
                        help="Figure file extension (default: pdf).")
    args = parser.parse_args()
    main(sOutputDirectory=args.output_directory,
         sFigureType=args.figure_type.lower())
