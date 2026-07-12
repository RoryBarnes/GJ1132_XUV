#!/usr/bin/env python3
"""
Infer GJ 1132's age from BOTH its rotation period AND its observed L_XUV/L_bol.

The rotation-only age posterior is supplied by the upstream Engle Age
Distribution step (rotation period ONLY, drawn from the Stage-1 hierarchical
refit with coefficient covariance + intrinsic scatter + asymmetric P_rot
error). This step folds in a SECOND, independent age constraint from the
observed L_XUV/L_bol via the X-UV relation's posterior-predictive: coefficients
are marginalized as nuisances and the inferred intrinsic scatter enters the
per-star likelihood. The two constraints are combined on a log-age grid; their
disagreement is quantified as a tension.

The rotation-only age is re-emitted unchanged so it is byte-identical to the
Engle Age Distribution output that feeds the Ribas branch: the L_XUV-informed
age is used ONLY in the EMD cumulative-flux comparison, never as the Ribas
age prior (that would double-use L_XUV).

Reference: Kelly (2007) ApJ 665, 1489 for the measurement-error framework.
"""

import argparse
import json

import numpy as np

I_SEED = 42
I_NUM_SAMPLES = 100000
I_GRID_POINTS = 2000

D_LOG_LXUV_LBOL = -4.26
D_LOG_LXUV_LBOL_SIGMA = 0.15
D_MAX_LOG_AGE = np.log10(13.0)
D_MIN_LOG_AGE = np.log10(0.1)


def fdaDrawChainRows(daChain, iCount):
    """Return iCount random rows drawn (with replacement) from a chain."""
    daIndices = np.random.randint(0, len(daChain), iCount)
    return daChain[daIndices]


def fdaXuvLogLikelihood(daTauGrid, daXuvChain):
    """Return the marginal L_XUV log-likelihood over a log-age grid."""
    daRows = fdaDrawChainRows(daXuvChain, len(daXuvChain))
    daLogLikelihood = np.empty(len(daTauGrid))
    for iIndex, dTau in enumerate(daTauGrid):
        daModelY = fdaHinge2d(daRows[:, :4], dTau)
        daSigmaSegment = np.where(dTau >= daRows[:, 3],
                                  np.exp(daRows[:, 5]), np.exp(daRows[:, 4]))
        daVariance = D_LOG_LXUV_LBOL_SIGMA ** 2 + daSigmaSegment ** 2
        daTerm = (-0.5 * (D_LOG_LXUV_LBOL - daModelY) ** 2 / daVariance
                  - 0.5 * np.log(daVariance))
        daLogLikelihood[iIndex] = fdLogMeanExp(daTerm)
    return daLogLikelihood


def fdaHinge2d(daCoefficientRows, dTau):
    """Evaluate the hinge at one age for many coefficient rows."""
    dA, dB = daCoefficientRows[:, 0], daCoefficientRows[:, 1]
    dC, dD = daCoefficientRows[:, 2], daCoefficientRows[:, 3]
    return dA * dTau + dB + dC * np.where(dTau >= dD, dTau - dD, 0.0)


def fdLogMeanExp(daValues):
    """Numerically stable log of the mean of exponentials."""
    dMax = np.max(daValues)
    return dMax + np.log(np.mean(np.exp(daValues - dMax)))


def fdaGaussianKdeDensity(daSamples, daGrid):
    """Return a Gaussian-KDE density estimate of daSamples on daGrid."""
    from scipy.stats import gaussian_kde
    return gaussian_kde(daSamples)(daGrid)


def fdaSampleFromGridDensity(daGrid, daDensity, iCount):
    """Draw iCount samples from a tabulated density via inverse-CDF sampling."""
    daDensity = np.clip(daDensity, 0, None)
    daCumulative = np.cumsum(daDensity)
    daCumulative /= daCumulative[-1]
    daUniform = np.random.uniform(0, 1, iCount)
    return np.interp(daUniform, daCumulative, daGrid)


def ftComputeInformedAges(daRotationTau, daXuvChain):
    """Return (informed_tau, rotation_tau, xuv_only_tau, grid, densities)."""
    daGrid = np.linspace(D_MIN_LOG_AGE, D_MAX_LOG_AGE, I_GRID_POINTS)
    daRotDensity = fdaGaussianKdeDensity(daRotationTau, daGrid)
    daXuvLogLike = fdaXuvLogLikelihood(daGrid, daXuvChain)
    daXuvLike = np.exp(daXuvLogLike - np.max(daXuvLogLike))
    daInformedDensity = daRotDensity * daXuvLike
    daInformedTau = fdaSampleFromGridDensity(daGrid, daInformedDensity,
                                             I_NUM_SAMPLES)
    daXuvOnlyTau = fdaSampleFromGridDensity(daGrid, daXuvLike, I_NUM_SAMPLES)
    return daInformedTau, daRotationTau, daXuvOnlyTau, daGrid, daInformedDensity


def fdComputeTensionSigma(daRotationTau, daXuvOnlyTau):
    """Return the disagreement between the two age constraints in sigma units."""
    dGap = abs(np.mean(daRotationTau) - np.mean(daXuvOnlyTau))
    dSpread = np.sqrt(np.var(daRotationTau) + np.var(daXuvOnlyTau))
    return dGap / dSpread


def fdictSummarizeAges(daTau, sLabel):
    """Return summary statistics for a log-age sample in Gyr."""
    daAgeGyr = 10 ** daTau
    return {
        "label": sLabel,
        "mean_log_age": float(np.mean(daTau)),
        "std_log_age": float(np.std(daTau)),
        "mean_age_gyr": float(np.mean(daAgeGyr)),
        "median_age_gyr": float(np.median(daAgeGyr)),
        "ci95_age_gyr": [float(np.percentile(daAgeGyr, 2.5)),
                         float(np.percentile(daAgeGyr, 97.5))],
    }


def fnSaveAgeYears(daTau, sPath):
    """Save log-age draws as ages in years (vconverge prior-file format)."""
    np.savetxt(sPath, 10 ** daTau * 1e9)


def fdaLoadRotationOnlyAgeYears(sPath):
    """Load the rotation-only age samples (years) from the Engle Age step."""
    return np.loadtxt(sPath)


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Infer GJ 1132 age from rotation and L_XUV/L_bol.")
    parser.add_argument("--rotation-only-ages", required=True,
                        help="Rotation-only age samples from the Engle Age "
                             "Distribution step (.txt, years).")
    parser.add_argument("--xuv-samples", required=True,
                        help="Stage-1 X-UV coefficient samples (.npy).")
    return parser.parse_args()


def main():
    """Infer GJ 1132's L_XUV-informed and rotation-only age posteriors."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daRotationAgeYears = fdaLoadRotationOnlyAgeYears(args.rotation_only_ages)
    daRotationTau = np.log10(daRotationAgeYears / 1e9)
    daXuvChain = np.load(args.xuv_samples)

    (daInformedTau, daRotationTau, daXuvOnlyTau,
     _, _) = ftComputeInformedAges(daRotationTau, daXuvChain)

    fnSaveAgeYears(daInformedTau, "lxuvInformedAgeSamples.txt")
    np.savetxt("rotationOnlyAgeSamples.txt", daRotationAgeYears)

    dictStats = {
        "rotation_only": fdictSummarizeAges(daRotationTau, "rotation-only"),
        "lxuv_informed": fdictSummarizeAges(daInformedTau, "L_XUV-informed"),
        "xuv_only": fdictSummarizeAges(daXuvOnlyTau, "L_XUV-only"),
        "tension_sigma": float(fdComputeTensionSigma(daRotationTau,
                                                     daXuvOnlyTau)),
    }
    with open("ageInferenceStats.json", "w") as fileHandle:
        json.dump(dictStats, fileHandle, indent=2)
    fnPrintSummary(dictStats)


def fnPrintSummary(dictStats):
    """Print a concise summary of the inferred age posteriors."""
    for sKey in ["rotation_only", "lxuv_informed", "xuv_only"]:
        dictEntry = dictStats[sKey]
        print(f"{dictEntry['label']:16s}: mean {dictEntry['mean_age_gyr']:.2f} "
              f"Gyr, 95% CI [{dictEntry['ci95_age_gyr'][0]:.2f}, "
              f"{dictEntry['ci95_age_gyr'][1]:.2f}] Gyr")
    print(f"Rotation-vs-L_XUV age tension: "
          f"{dictStats['tension_sigma']:.2f} sigma")


if __name__ == "__main__":
    main()
