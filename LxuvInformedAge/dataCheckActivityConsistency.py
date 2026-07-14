#!/usr/bin/env python3
"""
Check GJ 1132's observed X-ray activity against the refit activity-age relation.

This is a CONSISTENCY CHECK, not a joint age inference: the age comes from
rotation alone (upstream step), and the single observed log(L_X/L_bol) then
updates only GJ 1132's latent offset z within the relation's intrinsic
scatter (a hierarchical random-effects update). The rotation age is never
revised, so the gyrochronology calibration is not double-used. Everything is
computed in the NATIVE X-ray band; the MUSCLES conversion is applied only to
report band-consistent X-UV numbers.

Outputs: activityConsistency.json (summary) and zOffsetSamples.txt (the z
posterior, which downstream sweep priors may consume as the "informed"
scatter-offset prior under the persistent-offset hypothesis).
"""

import argparse
import json

import numpy as np

I_SEED = 42
D_SOLAR_LUMINOSITY = 3.846e33
D_LX_MEAN = 9.96e25
D_LX_STD = 2.95e25
D_LBOL_MEAN = 0.00477
D_LBOL_PLUS = 0.00036
D_LBOL_MINUS = 0.00026


def fdaSampleSplitNormal(dMean, dSigmaPlus, dSigmaMinus, iCount):
    """Draw split-normal samples with Wallis (2014) side masses."""
    daUniform = np.random.uniform(0, 1, iCount)
    baUpper = daUniform < dSigmaPlus / (dSigmaPlus + dSigmaMinus)
    daSamples = np.empty(iCount)
    daSamples[baUpper] = dMean + np.abs(
        np.random.normal(0, dSigmaPlus, int(np.sum(baUpper))))
    daSamples[~baUpper] = dMean - np.abs(
        np.random.normal(0, dSigmaMinus, int(np.sum(~baUpper))))
    return daSamples


def ftSampleObservedActivity(iCount):
    """Return draws of GJ 1132's observed log(L_X/L_bol)."""
    daLx = np.random.normal(D_LX_MEAN, D_LX_STD, iCount)
    daLbol = fdaSampleSplitNormal(D_LBOL_MEAN, D_LBOL_PLUS, D_LBOL_MINUS,
                                  iCount) * D_SOLAR_LUMINOSITY
    baValid = daLx > 0
    return np.log10(daLx[baValid] / daLbol[baValid])


def ftPredictActivity(daTau, daChain, dictSummary):
    """Return per-draw mean-line prediction and intrinsic scatter at daTau."""
    daRows = daChain[np.random.randint(0, len(daChain), len(daTau))]
    daMean = (daRows[:, 6] * daTau + daRows[:, 7]
              + daRows[:, 8] * np.clip(daTau - daRows[:, 9], 0.0, None))
    daSigma = np.exp(daRows[:, 10] + daRows[:, 11]
                     * (daTau - dictSummary["dPivotTau"])
                     / dictSummary["dScaleTau"])
    return daMean, daSigma


def fdaUpdateOffset(daMean, daSigma, daObserved):
    """Hierarchical random-effects update of GJ 1132's offset z."""
    dObservedMean = float(np.mean(daObserved))
    dObservedVariance = float(np.var(daObserved))
    daShrinkage = daSigma / (daSigma ** 2 + dObservedVariance)
    daZ = (dObservedMean - daMean) * daShrinkage
    daZ += np.random.normal(0, 1, len(daMean)) * np.sqrt(
        dObservedVariance / (daSigma ** 2 + dObservedVariance))
    return daZ


def fdictSummarize(daObserved, daMean, daSigma, daZ, dictConversion):
    """Build the consistency summary, native band plus X-UV report values."""
    dSlope = dictConversion["slope"]
    dIntercept = dictConversion["intercept"]
    return {
        "observed_log_lx_lbol": [float(np.mean(daObserved)),
                                 float(np.std(daObserved))],
        "predicted_log_lx_lbol": [float(np.mean(daMean)),
                                  float(np.std(daMean))],
        "sigma_int_at_rotation_age": float(np.mean(daSigma)),
        "z_offset": {"mean": float(np.mean(daZ)), "std": float(np.std(daZ)),
                     "prior": [0.0, 1.0]},
        "consistency_sigma": float(abs(np.mean(daObserved) - np.mean(daMean))
                                   / np.sqrt(np.var(daObserved)
                                             + np.var(daMean)
                                             + np.mean(daSigma) ** 2)),
        "observed_log_lxuv_lbol_via_muscles": [
            float(dSlope * np.mean(daObserved) + dIntercept),
            float(dSlope * np.std(daObserved))],
    }


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GJ 1132 activity-consistency check (native X-ray band).")
    parser.add_argument("--age-samples", required=True,
                        help="Rotation-only age samples (.txt, years).")
    parser.add_argument("--joint-chain", required=True,
                        help="Canonical joint-refit chain (.npy).")
    parser.add_argument("--fit-summary", required=True,
                        help="Matching joint-fit summary (.json).")
    parser.add_argument("--conversion-fit", required=True,
                        help="MUSCLES conversion fit (.json).")
    return parser.parse_args()


def main():
    """Run the consistency check and persist its products."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daTau = np.log10(np.loadtxt(args.age_samples) / 1e9)
    daChain = np.load(args.joint_chain)
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    with open(args.conversion_fit) as fileHandle:
        dictConversion = json.load(
            fileHandle)["primary_fit_all_targets_rosat_band"]
    daObserved = ftSampleObservedActivity(len(daTau))
    iCount = min(len(daObserved), len(daTau))
    daMean, daSigma = ftPredictActivity(daTau[:iCount], daChain, dictSummary)
    daZ = fdaUpdateOffset(daMean, daSigma, daObserved[:iCount])
    np.savetxt("zOffsetSamples.txt", daZ)
    dictOut = fdictSummarize(daObserved[:iCount], daMean, daSigma, daZ,
                             dictConversion)
    with open("activityConsistency.json", "w") as fileHandle:
        json.dump(dictOut, fileHandle, indent=2)
    print(f"z offset: {dictOut['z_offset']['mean']:+.2f} +/- "
          f"{dictOut['z_offset']['std']:.2f} "
          f"(consistency {dictOut['consistency_sigma']:.2f} sigma)")


if __name__ == "__main__":
    main()
