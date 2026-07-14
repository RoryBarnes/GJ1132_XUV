#!/usr/bin/env python3
"""
Compute GJ 1132's rotation-only gyrochronology age distribution.

Draws from the joint hierarchical refit posterior (canonical variant: halo
subdwarfs included, paper-faithful M2.5-6.5 composition). Each draw takes one
posterior row (rotation hinge coefficients with full covariance), one rotation
period from the split-normal measurement distribution (Wallis 2014 side
masses), and one deviate from that row's log-linear intrinsic-scatter law
evaluated at the drawn period. Rotation period ONLY: the observed L_XUV/L_bol
is deliberately NOT used here, so this age can serve as the Ribas branch's
age prior without double-counting L_XUV.

Reference: Kelly (2007) ApJ 665, 1489 for the measurement-error framework.
"""

import argparse
import json

import numpy as np

I_SEED = 42
I_NUM_SAMPLES = 100000
D_PROT_MEAN = 122.3
D_PROT_PLUS = 6.0
D_PROT_MINUS = 5.0
D_MAX_LOG_AGE = np.log10(13.0)


def fdaSampleSplitNormal(dMean, dSigmaPlus, dSigmaMinus, iCount):
    """Draw samples from a split-normal (asymmetric Gaussian) distribution.

    Side masses follow Wallis (2014): P(upper) = sigma+ / (sigma+ + sigma-),
    which keeps the density continuous at the mode.
    """
    daUniform = np.random.uniform(0, 1, iCount)
    baUpper = daUniform < dSigmaPlus / (dSigmaPlus + dSigmaMinus)
    daSamples = np.empty(iCount)
    daSamples[baUpper] = dMean + np.abs(
        np.random.normal(0, dSigmaPlus, int(np.sum(baUpper))))
    daSamples[~baUpper] = dMean - np.abs(
        np.random.normal(0, dSigmaMinus, int(np.sum(~baUpper))))
    return daSamples


def fdaDrawChainRows(daChain, iCount):
    """Return iCount rows drawn with replacement from a posterior chain."""
    daIndices = np.random.randint(0, len(daChain), iCount)
    return daChain[daIndices]


def fdaEvaluateHinge(daCoefficientRows, daX):
    """Evaluate the continuous two-segment hinge, one coefficient row per point."""
    dA, dB = daCoefficientRows[:, 0], daCoefficientRows[:, 1]
    dC, dD = daCoefficientRows[:, 2], daCoefficientRows[:, 3]
    return dA * daX + dB + dC * np.where(daX >= dD, daX - dD, 0.0)


def fdaEvaluateScatterLaw(daRows, daProt, dictSummary):
    """Evaluate each row's log-linear intrinsic-scatter law at each period."""
    daLogSigma = (daRows[:, 4] + daRows[:, 5]
                  * (daProt - dictSummary["dPivotProt"])
                  / dictSummary["dScaleProt"])
    return np.exp(daLogSigma)


def fdaComputeRotationOnlyAges(daChain, dictSummary):
    """Return rotation-only log-age draws: covariance, scatter law, P_rot."""
    daRows = fdaDrawChainRows(daChain, I_NUM_SAMPLES)
    daProt = fdaSampleSplitNormal(D_PROT_MEAN, D_PROT_PLUS, D_PROT_MINUS,
                                  I_NUM_SAMPLES)
    daModelTau = fdaEvaluateHinge(daRows[:, 0:4], daProt)
    daSigma = fdaEvaluateScatterLaw(daRows, daProt, dictSummary)
    daTau = daModelTau + np.random.normal(0, 1, I_NUM_SAMPLES) * daSigma
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
    parser.add_argument("--joint-chain", required=True,
                        help="Canonical joint-refit hyperparameter chain "
                             "(.npy, 12 columns).")
    parser.add_argument("--fit-summary", required=True,
                        help="Matching joint-fit summary (.json) carrying the "
                             "scatter-law pivots.")
    return parser.parse_args()


def main():
    """Compute and save the rotation-only age distribution."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daChain = np.load(args.joint_chain)
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    daLogAge = fdaComputeRotationOnlyAges(daChain, dictSummary)
    fnPrintStatistics(daLogAge)
    fnSaveAgeSamples(daLogAge, "age_samples.txt")


if __name__ == "__main__":
    main()
