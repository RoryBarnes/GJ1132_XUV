#!/usr/bin/env python3
"""
Compute GJ 1132's rotation-only gyrochronology age distribution.

Draws from the Stage-1 hierarchical refit posterior of the Engle & Guinan
rotation relation (coefficient samples carry the full covariance), folds in the
rotation relation's intrinsic scatter and the asymmetric P_rot measurement
error, and saves the resulting age distribution. Rotation period ONLY: the
observed L_XUV/L_bol is deliberately NOT used here so this age can serve as the
Ribas branch's age prior without double-counting L_XUV.

Reference: Kelly (2007) ApJ 665, 1489 for the measurement-error framework.
"""

import argparse

import numpy as np

I_SEED = 42
I_NUM_SAMPLES = 100000
D_PROT_MEAN = 122.3
D_PROT_PLUS = 6.0
D_PROT_MINUS = 5.0
D_MAX_LOG_AGE = np.log10(13.0)


def fdaSampleSplitNormal(dMean, dSigmaPlus, dSigmaMinus, iCount):
    """Draw samples from a split-normal (asymmetric Gaussian) distribution."""
    daUniform = np.random.uniform(0, 1, iCount)
    baUpper = daUniform > 0.5
    daSamples = np.empty(iCount)
    daSamples[baUpper] = dMean + np.abs(
        np.random.normal(0, dSigmaPlus, int(np.sum(baUpper))))
    daSamples[~baUpper] = dMean - np.abs(
        np.random.normal(0, dSigmaMinus, int(np.sum(~baUpper))))
    return daSamples


def fdaDrawChainRows(daChain, iCount):
    """Return iCount rows drawn with replacement from a coefficient chain."""
    daIndices = np.random.randint(0, len(daChain), iCount)
    return daChain[daIndices]


def fdaEvaluateHinge(daCoefficientRows, daX):
    """Evaluate the continuous two-segment hinge, one coefficient row per point."""
    dA, dB = daCoefficientRows[:, 0], daCoefficientRows[:, 1]
    dC, dD = daCoefficientRows[:, 2], daCoefficientRows[:, 3]
    return dA * daX + dB + dC * np.where(daX >= dD, daX - dD, 0.0)


def fdaComputeRotationOnlyAges(daRotChain):
    """Return rotation-only log-age draws, folding covariance, sigma_int, P_rot."""
    daRows = fdaDrawChainRows(daRotChain, I_NUM_SAMPLES)
    daProt = fdaSampleSplitNormal(D_PROT_MEAN, D_PROT_PLUS, D_PROT_MINUS,
                                  I_NUM_SAMPLES)
    daModelTau = fdaEvaluateHinge(daRows[:, :4], daProt)
    daSigmaOld = np.exp(daRows[:, 5])
    daTau = daModelTau + np.random.normal(0, daSigmaOld)
    return daTau[daTau <= D_MAX_LOG_AGE]


def fnSaveAgeSamples(daLogAge, sOutputFile):
    """Convert log-age to years and save to file."""
    daAge = 10 ** daLogAge * 1e9
    np.savetxt(sOutputFile, daAge)
    print(f"Saved {len(daAge):,} samples to '{sOutputFile}' "
          f"({np.min(daAge) / 1e9:.2f} - {np.max(daAge) / 1e9:.2f} Gyr)")


def fnPrintStatistics(daLogAge):
    """Print summary statistics of the rotation-only age distribution."""
    daAgeGyr = 10 ** daLogAge
    print(f"Rotation-only age: mean {np.mean(daAgeGyr):.2f} Gyr, "
          f"median {np.median(daAgeGyr):.2f} Gyr")
    print(f"mean log-age {np.mean(daLogAge):.4f} +/- {np.std(daLogAge):.4f}")
    print(f"95% CI [{np.percentile(daAgeGyr, 2.5):.2f}, "
          f"{np.percentile(daAgeGyr, 97.5):.2f}] Gyr")


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute GJ 1132 rotation-only gyrochronology age.")
    parser.add_argument("--rotation-samples", required=True,
                        help="Stage-1 rotation coefficient samples (.npy).")
    return parser.parse_args()


def main():
    """Compute and save the rotation-only age distribution."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daRotChain = np.load(args.rotation_samples)
    daLogAge = fdaComputeRotationOnlyAges(daRotChain)
    fnPrintStatistics(daLogAge)
    fnSaveAgeSamples(daLogAge, "age_samples.txt")


if __name__ == "__main__":
    main()
