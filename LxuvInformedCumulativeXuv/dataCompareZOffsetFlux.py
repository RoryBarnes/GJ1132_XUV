#!/usr/bin/env python3
"""
Compare cumulative XUV flux on GJ 1132 b under two scatter-offset priors.

Both branches use the IDENTICAL rotation-only age prior; the only difference
is where GJ 1132 sits within the activity relation's intrinsic scatter:
population prior z ~ N(0, 1) (no L_XUV measurement) versus the informed
posterior z | observed L_X from the Activity Consistency Check. Comparing the
two flux distributions therefore isolates the value of a single host-star
X-ray measurement for a habitability-relevant quantity — the community
anchor point. Both branches also carry the coefficient covariance and the
MUSCLES conversion uncertainty; models with "Barnes" include the flare
contribution. The constant-offset z treatment is the bracket approximation
until the vplanet scatter-law option lands; sigma_int is evaluated at
GJ 1132's rotation age (the conservative reference).

vconverge is the propagation tool here, not the inference engine.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.cumulativeXuv import (ftGatherFluxes, fdictLoadConvergedJson,
                                 daExtractFluxValues)
from utils.englePriorTable import fnWriteEnglePriorTable

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
S_CONVERGED = "output/Converged_Param_Dictionary.json"
S_BACKUP = ".Converged_Param_Dictionary.json.bak"
S_PRIOR_TABLE = "engle_xuv_priors.txt"
D_TAU_REFERENCE = 0.85

DICT_MODELS = {
    "EngleRotationOnly": ("population", "cumulativeXuvFluxSamplesZPopulation.txt"),
    "EngleLxuvInformed": ("informed", "cumulativeXuvFluxSamplesZInformed.txt"),
    "EngleBarnesRotationOnly": ("population",
                                "cumulativeXuvFluxSamplesFlaresZPopulation.txt"),
    "EngleBarnesLxuvInformed": ("informed",
                                "cumulativeXuvFluxSamplesFlaresZInformed.txt"),
}


def fnStageModelInputs(sModelDirectory, args, sZMode):
    """Copy the age prior and write the z-mode prior table for one model."""
    shutil.copy2(args.age_samples,
                 os.path.join(sModelDirectory, "age_samples.txt"))
    fnWriteEnglePriorTable(
        args.joint_chain, args.fit_summary, args.conversion_fit,
        os.path.join(sModelDirectory, S_PRIOR_TABLE), sZMode=sZMode,
        sZOffsetsPath=args.z_offsets, dTauReference=D_TAU_REFERENCE)


def fbRunVconverge(sModelDirectory):
    """Run vconverge in a model directory, returning True on success."""
    print(f"[vaib] {sModelDirectory}: running vconverge", flush=True)
    result = subprocess.run(["vconverge", "vconverge.in"], cwd=sModelDirectory)
    return result.returncode == 0


def fbProcessModel(sModelDirectory, args, sZMode):
    """Stage inputs, run vconverge, and manage the backup for one model."""
    fnStageModelInputs(sModelDirectory, args, sZMode)
    sSource = os.path.join(sModelDirectory, S_CONVERGED)
    if os.path.exists(sSource):
        shutil.copy2(sSource, os.path.join(sModelDirectory, S_BACKUP))
    bSuccess = fbRunVconverge(sModelDirectory)
    if bSuccess and os.path.exists(sSource):
        return True
    sBackup = os.path.join(sModelDirectory, S_BACKUP)
    if os.path.exists(sBackup):
        shutil.copy2(sBackup, sSource)
    return False


def fdictSummarizeFlux(sModelDirectory, sOutputSamplesFile):
    """Extract flux, save samples, and return summary statistics."""
    sConvergedPath = os.path.join(sModelDirectory, S_CONVERGED)
    _, _, dMean, dLower, dUpper = ftGatherFluxes(sConvergedPath)
    daFlux = daExtractFluxValues(fdictLoadConvergedJson(sConvergedPath))
    np.savetxt(os.path.join(S_DIRECTORY, sOutputSamplesFile), daFlux)
    return {"count": int(len(daFlux)), "mean": float(dMean),
            "ci95": [float(dLower), float(dUpper)]}


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cumulative XUV flux under population vs informed z.")
    parser.add_argument("--age-samples", required=True,
                        help="Rotation-only age samples (.txt, years).")
    parser.add_argument("--joint-chain", required=True)
    parser.add_argument("--fit-summary", required=True)
    parser.add_argument("--conversion-fit", required=True)
    parser.add_argument("--z-offsets", required=True,
                        help="Informed z posterior samples (.txt).")
    return parser.parse_args()


def main():
    """Run all four model variants and summarize the comparison."""
    args = ftParseArguments()
    os.chdir(S_DIRECTORY)
    dictStats = {}
    for sModelDirectory, (sZMode, sSamplesFile) in DICT_MODELS.items():
        if not fbProcessModel(sModelDirectory, args, sZMode):
            sys.exit(1)
        dictStats[sModelDirectory] = fdictSummarizeFlux(sModelDirectory,
                                                        sSamplesFile)
    with open("cumulativeXuvStats.json", "w") as fileHandle:
        json.dump(dictStats, fileHandle, indent=2)
    for sKey, dictEntry in dictStats.items():
        print(f"{sKey:26s}: mean {dictEntry['mean']:.0f}, 95% CI "
              f"[{dictEntry['ci95'][0]:.0f}, {dictEntry['ci95'][1]:.0f}]")


if __name__ == "__main__":
    main()
