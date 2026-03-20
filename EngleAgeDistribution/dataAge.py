import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import vplot

I_NUM_BINS = 50
D_FIG_SIZE_X = 3.25
D_FIG_SIZE_Y = 3
D_MAX_LOG_AGE = np.log10(13)


def ftComputeAgeDistribution(daA, daB, daC, daD, daRotationPeriod,
                              iNumSamples=100000):
    """Return (daLogAge, dMean, dStdDev, daConfidenceInterval)."""
    daASamples = np.random.normal(daA[0], daA[1], iNumSamples)
    daBSamples = np.random.normal(daB[0], daB[1], iNumSamples)
    daCSamples = np.random.normal(daC[0], daC[1], iNumSamples)
    daDSamples = np.random.normal(daD[0], daD[1], iNumSamples)
    daRotSamples = np.random.normal(
        daRotationPeriod[0], daRotationPeriod[1], iNumSamples)

    daLogAge = (daASamples * daRotSamples + daBSamples
                + daCSamples * (daRotSamples - daDSamples))
    daLogAge = daLogAge[daLogAge <= D_MAX_LOG_AGE]

    print(f"Samples after filtering: {len(daLogAge):,} "
          f"out of {iNumSamples:,} ({100 * len(daLogAge) / iNumSamples:.1f}%)")

    dMean = np.mean(daLogAge)
    dStdDev = np.std(daLogAge)
    daCI = np.percentile(daLogAge, [2.5, 97.5])
    return daLogAge, dMean, dStdDev, daCI


def fdComputeAnalyticalMeanAge(daA, daB, daC, daD, daRotationPeriod):
    """Return the analytical mean log(age) for comparison."""
    return ((daA[0] + daC[0]) * daRotationPeriod[0]
            + daB[0] - daC[0] * daD[0])


def fnPlotNormalizedHistogram(daLogAge, sOutputPath):
    """Save a normalized histogram of the age distribution."""
    plt.figure(figsize=(D_FIG_SIZE_X, D_FIG_SIZE_Y))
    daAge = 10**daLogAge
    daCounts, daBinEdges = np.histogram(daAge, bins=I_NUM_BINS)
    daFractions = daCounts / len(daAge)
    plt.step(daBinEdges[:-1], daFractions, where='mid', color='k')
    plt.xlabel('Age [Gyr]', fontsize=12)
    plt.ylabel('Fraction', fontsize=12)
    plt.xlim(0, 13)
    plt.ylim(0, 0.04)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(sOutputPath, dpi=300)
    plt.close()


def fnPrintStatistics(daLogAge, dMean, dStdDev, daCI, dAnalytical):
    """Print Monte Carlo and analytical results."""
    print(f"\nMonte Carlo Results ({len(daLogAge):,} samples):")
    print(f"Best fit (mean): {dMean:.4f}")
    print(f"Uncertainty (std): {dStdDev:.4f}")
    print(f"95% CI: [{daCI[0]:.4f}, {daCI[1]:.4f}]")
    print(f"Analytical mean: {dAnalytical:.4f}")
    print(f"MC-Analytical difference: {abs(dMean - dAnalytical):.6f}")
    print(f"Median: {np.median(daLogAge):.4f}")
    print(f"Skewness: {stats.skew(daLogAge):.4f}")
    print(f"Kurtosis: {stats.kurtosis(daLogAge):.4f}")


def fnSaveAgeSamples(daLogAge, sOutputFile):
    """Convert log-age to years and save to file."""
    daAge = 10**daLogAge * 1e9
    np.savetxt(sOutputFile, daAge)
    print(f"\nSaved to '{sOutputFile}' "
          f"({np.min(daAge) / 1e9:.2f} - {np.max(daAge) / 1e9:.2f} Gyr)")


if __name__ == "__main__":
    sPlotFile = sys.argv[1] if len(sys.argv) > 1 else "EngleAgeHist.pdf"

    daA = (0.0251, 0.0018)
    daB = (-0.1615, 0.0303)
    daC = (-0.0212, 0.0018)
    daD = (25.45, 1.9079)
    daRotationPeriod = (122, 5.5)

    daLogAge, dMean, dStdDev, daCI = ftComputeAgeDistribution(
        daA, daB, daC, daD, daRotationPeriod)
    dAnalytical = fdComputeAnalyticalMeanAge(daA, daB, daC, daD, daRotationPeriod)

    fnPrintStatistics(daLogAge, dMean, dStdDev, daCI, dAnalytical)
    fnPlotNormalizedHistogram(daLogAge, sPlotFile)
    fnSaveAgeSamples(daLogAge, 'age_samples.txt')
