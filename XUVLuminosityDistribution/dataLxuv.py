#!/usr/bin/env python3
"""
Calculate XUV luminosity distribution for GJ 1132 using X-ray observations.

Performs Monte Carlo uncertainty propagation using observed X-ray luminosity,
EUV-to-X-ray scaling relation, and bolometric luminosity with asymmetric errors.
"""

import json

import numpy as np
from scipy import stats

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


def fnProcessDistribution(daSamples, sLabel, sSampleFile):
    """Compute stats, print, and save samples for a distribution."""
    dMean, dStd, daCI = ftComputeStatistics(daSamples)
    fnPrintBriefStatistics(sLabel, dMean, dStd, daCI)
    fnSaveSamples(daSamples, sSampleFile)


def main():
    """Generate all L_XUV distribution data files and statistics."""
    np.random.seed(42)

    daLxuvErgs = daCalculateLxuvDistribution()
    daLxuvLsun = daConvertToLsun(daLxuvErgs)
    dMean, dStd, daCI = ftComputeStatistics(daLxuvLsun)
    fnPrintLxuvStatistics(daLxuvLsun, dMean, dStd, daCI)
    fnSaveSamples(daLxuvLsun, 'lxuv_samples.txt')

    daRatio = daCalculateLxuvLbolDistribution(daLxuvLsun)
    fnProcessDistribution(daRatio, "L_XUV/L_bol", 'lxuv_lbol_samples.txt')

    daLogRatio = np.log10(daRatio)
    fnProcessDistribution(
        daLogRatio, "log10(L_XUV/L_bol)", 'log_lxuv_lbol_samples.txt')

    dLogRatioMean, dLogRatioStd, _ = ftComputeStatistics(daLogRatio)
    fnWriteConstraintsJson(dLogRatioMean, dLogRatioStd,
                           'lxuv_constraints.json')

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
