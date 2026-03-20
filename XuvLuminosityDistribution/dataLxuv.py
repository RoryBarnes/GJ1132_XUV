#!/usr/bin/env python3
"""
Calculate XUV luminosity distribution for GJ 1132 using X-ray observations.

Performs Monte Carlo uncertainty propagation using observed X-ray luminosity,
EUV-to-X-ray scaling relation, and bolometric luminosity with asymmetric errors.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import vplot

D_L_SUN = 3.846e33

D_L_X_MEAN = 9.96e25
D_L_X_STD = 2.95e25
D_SLOPE_MEAN = 0.821
D_SLOPE_STD = 0.041
D_INTERCEPT_MEAN = 28.16
D_INTERCEPT_STD = 0.05
D_C_OFFSET = 27.44

I_NUM_SAMPLES = 10000

D_L_BOL_MEAN = 0.00477
D_L_BOL_SIGMA_PLUS = 0.00036
D_L_BOL_SIGMA_MINUS = 0.00026

D_FIG_SIZE_X = 3.25
D_FIG_SIZE_Y = 3
I_NUM_BINS = 50


def ftSampleAndFilterParameters():
    """Sample L_X, slope, and intercept; filter out unphysical L_X < 0."""
    daLxSamples = np.random.normal(D_L_X_MEAN, D_L_X_STD, I_NUM_SAMPLES)
    daSlopeSamples = np.random.normal(D_SLOPE_MEAN, D_SLOPE_STD, I_NUM_SAMPLES)
    daInterceptSamples = np.random.normal(D_INTERCEPT_MEAN, D_INTERCEPT_STD,
                                          I_NUM_SAMPLES)
    baValid = daLxSamples > 0
    print(f"Samples after filtering (L_X > 0): {np.sum(baValid):,} "
          f"out of {I_NUM_SAMPLES:,}")
    return daLxSamples[baValid], daSlopeSamples[baValid], daInterceptSamples[baValid]


def daCalculateLxuvDistribution():
    """Calculate L_XUV distribution via Monte Carlo sampling (erg/s)."""
    daLx, daSlope, daIntercept = ftSampleAndFilterParameters()
    daC = np.log10(daLx) - D_C_OFFSET
    daLogLeuv = daSlope * daC + daIntercept
    return daLx + 10.0 ** daLogLeuv


def daConvertToLsun(daLxuvErgs):
    """Convert L_XUV from erg/s to solar luminosity units."""
    return daLxuvErgs / D_L_SUN


def ftComputeStatistics(daSamples):
    """Return (dMean, dStd, daCI) for a sample array."""
    dMean = np.mean(daSamples)
    dStd = np.std(daSamples)
    daCI = np.percentile(daSamples, [2.5, 97.5])
    return dMean, dStd, daCI


def fnPrintLxuvStatistics(daSamples, dMean, dStd, daCI):
    """Print full L_XUV statistics including normality assessment."""
    print("=" * 70)
    print("XUV Luminosity Distribution Statistics")
    print("=" * 70)
    print(f"Monte Carlo Results ({len(daSamples):,} samples):")
    print(f"  Mean:     {dMean:.4e} LSUN ({dMean * D_L_SUN:.4e} erg/s)")
    print(f"  Std:      {dStd:.4e} LSUN ({dStd * D_L_SUN:.4e} erg/s)")
    print(f"  95% CI:   [{daCI[0]:.4e}, {daCI[1]:.4e}] LSUN")
    fnPrintNormalityAssessment(daSamples)
    print("=" * 70)


def fnPrintNormalityAssessment(daSamples):
    """Print skewness, kurtosis, and Shapiro-Wilk test results."""
    dSkewness = stats.skew(daSamples)
    dKurtosis = stats.kurtosis(daSamples)
    _, dShapiroP = stats.shapiro(daSamples[:5000])
    print(f"  Median:   {np.median(daSamples):.4e} LSUN")
    print(f"  Skewness: {dSkewness:.4f}")
    print(f"  Kurtosis: {dKurtosis:.4f}")
    sNormal = "Yes (p > 0.05)" if dShapiroP > 0.05 else "No (p < 0.05)"
    print(f"  Shapiro-Wilk p-value: {dShapiroP:.6f} — {sNormal}")


def fnPrintBriefStatistics(sLabel, dMean, dStd, daCI):
    """Print a short summary line for a derived distribution."""
    print(f"{sLabel}:")
    print(f"  Mean: {dMean:.4e}  Std: {dStd:.4e}  "
          f"95% CI: [{daCI[0]:.4e}, {daCI[1]:.4e}]")


def fnPlotDistributionPanel(ax, daSamples, dMean, daCI, sXlabel, sTitle):
    """Plot a histogram panel with mean and CI lines."""
    ax.hist(daSamples, bins=100, density=True, alpha=0.7,
            color='skyblue', edgecolor='black')
    ax.axvline(dMean, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {dMean:.4e}')
    ax.axvline(daCI[0], color='orange', linestyle=':', linewidth=2,
               label=f'95% CI: [{daCI[0]:.4e}, {daCI[1]:.4e}]')
    ax.axvline(daCI[1], color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel(sXlabel, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(sTitle, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)


def fnPlotDiagnosticFigure(daSamples, dMean, daCI):
    """Create two-panel diagnostic plot: histogram + Q-Q plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6))
    fnPlotDistributionPanel(ax1, daSamples, dMean, daCI,
                            r'$L_{XUV}$ [$L_\odot$]',
                            'GJ 1132 XUV Luminosity Distribution')
    stats.probplot(daSamples, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Normality Assessment', fontsize=12,
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lxuv_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def fnSaveSamples(daSamples, sFilename):
    """Save samples to a text file."""
    np.savetxt(sFilename, daSamples)
    print(f"Saved {len(daSamples):,} samples to: {sFilename}")


def fnWriteConstraintsJson(dMean, dStd, sFilename):
    """Write log10(Lxuv/Lbol) constraint to a JSON file."""
    dictConstraint = {"dMean": round(dMean, 4), "dStd": round(dStd, 4)}
    with open(sFilename, "w") as fileHandle:
        json.dump(dictConstraint, fileHandle, indent=2)
    print(f"Saved constraint to: {sFilename}")


def daSampleAsymmetricNormal(dMean, dSigmaPlus, dSigmaMinus, iNumSamples):
    """Sample from a split-normal distribution with asymmetric errors."""
    daUniform = np.random.uniform(0, 1, iNumSamples)
    daSamples = np.zeros(iNumSamples)
    baMaskUpper = daUniform > 0.5
    iUpper = np.sum(baMaskUpper)
    daSamples[baMaskUpper] = dMean + np.abs(
        np.random.normal(0, dSigmaPlus, iUpper))
    daSamples[~baMaskUpper] = dMean - np.abs(
        np.random.normal(0, dSigmaMinus, iNumSamples - iUpper))
    return daSamples


def daCalculateLxuvLbolDistribution(daLxuvLsun):
    """Calculate L_XUV/L_bol ratio distribution."""
    daLbolSamples = daSampleAsymmetricNormal(
        D_L_BOL_MEAN, D_L_BOL_SIGMA_PLUS, D_L_BOL_SIGMA_MINUS,
        len(daLxuvLsun))
    return daLxuvLsun / daLbolSamples


def fnPlotStepHistogram(daSamples, dMean, dStd, sXlabel, dXscale=1.0):
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
    fnPlotStepHistogram(daSamples, dMean, dStd, sXlabel, dXscale)
    fnFormatNormalizedHistogram(sXlabel)
    plt.savefig(sFilename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to: {sFilename}")


def fsResolvePath(sFilename, sOutputDirectory=None):
    """Resolve an output filename with an optional output directory."""
    if sOutputDirectory:
        return os.path.join(sOutputDirectory, sFilename)
    return sFilename


def fnProcessDistribution(daSamples, sLabel, sSampleFile, sXlabel,
                          sHistFile, dXscale=1.0):
    """Compute stats, print, save samples, and plot histogram for a distribution."""
    dMean, dStd, daCI = ftComputeStatistics(daSamples)
    fnPrintBriefStatistics(sLabel, dMean, dStd, daCI)
    fnSaveSamples(daSamples, sSampleFile)
    fnPlotNormalizedHistogram(daSamples, dMean, dStd, sXlabel, sHistFile,
                              dXscale)


def main(sOutputDirectory=None, sFigureType="pdf"):
    """Generate all L_XUV distribution figures and statistics."""
    np.random.seed(42)

    daLxuvErgs = daCalculateLxuvDistribution()
    daLxuvLsun = daConvertToLsun(daLxuvErgs)
    dMean, dStd, daCI = ftComputeStatistics(daLxuvLsun)
    fnPrintLxuvStatistics(daLxuvLsun, dMean, dStd, daCI)
    fnPlotDiagnosticFigure(daLxuvLsun, dMean, daCI)
    fnSaveSamples(daLxuvLsun, 'lxuv_samples.txt')

    fnPlotNormalizedHistogram(
        daLxuvLsun, dMean, dStd,
        sXlabel=r'$L_{XUV}$ [$10^{-7} L_\odot$]',
        sFilename=fsResolvePath(f'lxuv_hist.{sFigureType}', sOutputDirectory),
        dXscale=1e7)

    daRatio = daCalculateLxuvLbolDistribution(daLxuvLsun)
    fnProcessDistribution(
        daRatio, "L_XUV/L_bol", 'lxuv_lbol_samples.txt',
        r'$L_{XUV} / L_{bol}$',
        fsResolvePath(f'lxuv_lbol_hist.{sFigureType}', sOutputDirectory))

    daLogRatio = np.log10(daRatio)
    fnProcessDistribution(
        daLogRatio, "log10(L_XUV/L_bol)", 'log_lxuv_lbol_samples.txt',
        r'$\log_{10}(L_{XUV} / L_{bol})$',
        fsResolvePath(f'log_lxuv_lbol_hist.{sFigureType}', sOutputDirectory))

    dLogRatioMean, dLogRatioStd, _ = ftComputeStatistics(daLogRatio)
    fnWriteConstraintsJson(dLogRatioMean, dLogRatioStd,
                           'lxuv_constraints.json')

    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="GJ 1132 XUV luminosity distribution calculator.")
    parser.add_argument('--output-directory', metavar='PATH',
                        help="Directory for output figure files.")
    parser.add_argument('--figure-type', default='pdf',
                        help="Figure file extension (default: pdf).")
    args = parser.parse_args()
    main(sOutputDirectory=args.output_directory,
         sFigureType=args.figure_type.lower())
