#!/usr/bin/env python3
"""
Propagate GJ 1132 age posteriors to cumulative XUV flux on planet b via vconverge.

Runs the Engle EMD forward model (vplanet, sampling the Table-2 XUV coefficient
priors) over two age priors from the Stage-2 inference: the rotation-only age
PDF and the L_XUV-informed age PDF. Using a common propagation isolates the
effect of folding the observed L_XUV/L_bol into the age. vconverge is the final
propagation tool here, not the inference engine.
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

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
S_CONVERGED = "output/Converged_Param_Dictionary.json"
S_BACKUP = ".Converged_Param_Dictionary.json.bak"


def fnBackupConverged(sModelDirectory):
    """Back up an existing converged output before a rerun."""
    sSource = os.path.join(sModelDirectory, S_CONVERGED)
    if os.path.exists(sSource):
        shutil.copy2(sSource, os.path.join(sModelDirectory, S_BACKUP))


def fnRestoreConverged(sModelDirectory):
    """Restore the converged output from backup after a failed run."""
    sBackup = os.path.join(sModelDirectory, S_BACKUP)
    if os.path.exists(sBackup):
        shutil.copy2(sBackup, os.path.join(sModelDirectory, S_CONVERGED))


def fnCopyAgePrior(sModelDirectory, sAgeSamplesPath):
    """Copy an age-sample prior file into a vconverge model directory."""
    shutil.copy2(sAgeSamplesPath,
                 os.path.join(sModelDirectory, "age_samples.txt"))


def fbRunVconverge(sModelDirectory):
    """Run vconverge in a model directory, returning True on success."""
    print(f"[vaib] {sModelDirectory}: running vconverge")
    result = subprocess.run(["vconverge", "vconverge.in"], cwd=sModelDirectory)
    return result.returncode == 0


def fnRequireConverged(sModelDirectory):
    """Raise if vconverge did not produce its converged-parameter output."""
    sSource = os.path.join(sModelDirectory, S_CONVERGED)
    if not os.path.exists(sSource):
        raise FileNotFoundError(f"vconverge produced no output: {sSource}")


def fbProcessModel(sModelDirectory, sAgeSamplesPath):
    """Copy the age prior, run vconverge, and manage the backup for one model."""
    fnCopyAgePrior(sModelDirectory, sAgeSamplesPath)
    fnBackupConverged(sModelDirectory)
    bSuccess = fbRunVconverge(sModelDirectory)
    if bSuccess:
        fnRequireConverged(sModelDirectory)
    else:
        print(f"[vaib] {sModelDirectory}: vconverge failed, restoring backup")
        fnRestoreConverged(sModelDirectory)
    return bSuccess


def fdictSummarizeFlux(sModelDirectory, sOutputSamplesFile):
    """Extract flux, save samples, and return summary statistics for a model."""
    sConvergedPath = os.path.join(sModelDirectory, S_CONVERGED)
    _, _, dMean, dLower, dUpper = ftGatherFluxes(sConvergedPath)
    daFlux = daExtractFluxValues(fdictLoadConvergedJson(sConvergedPath))
    np.savetxt(os.path.join(S_DIRECTORY, sOutputSamplesFile), daFlux)
    return {
        "count": int(len(daFlux)),
        "mean": float(dMean),
        "ci95": [float(dLower), float(dUpper)],
    }


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Propagate GJ 1132 age posteriors to cumulative XUV flux.")
    parser.add_argument("--lxuv-informed-ages", required=True,
                        help="L_XUV-informed age samples (.txt, years).")
    parser.add_argument("--rotation-only-ages", required=True,
                        help="Rotation-only age samples (.txt, years).")
    return parser.parse_args()


def main():
    """Run vconverge for both age priors and summarize the flux distributions."""
    args = ftParseArguments()
    os.chdir(S_DIRECTORY)
    dictModels = {
        "EngleLxuvInformed": (args.lxuv_informed_ages,
                              "cumulativeXuvFluxSamplesInformed.txt"),
        "EngleRotationOnly": (args.rotation_only_ages,
                              "cumulativeXuvFluxSamplesRotationOnly.txt"),
        "EngleBarnesLxuvInformed": (args.lxuv_informed_ages,
                                    "cumulativeXuvFluxSamplesFlaresInformed.txt"),
        "EngleBarnesRotationOnly": (args.rotation_only_ages,
                                    "cumulativeXuvFluxSamplesFlaresRotationOnly.txt"),
    }
    dictStats = {}
    for sModelDirectory, (sAgePath, sSamplesFile) in dictModels.items():
        if not fbProcessModel(sModelDirectory, sAgePath):
            sys.exit(1)
        dictStats[sModelDirectory] = fdictSummarizeFlux(
            sModelDirectory, sSamplesFile)
    dictStats["published_emd_only"] = {"mean": 402, "ci95": [241, 571]}
    dictStats["published_emd_flares"] = {"mean": 484, "ci95": [291, 671]}
    with open("cumulativeXuvStats.json", "w") as fileHandle:
        json.dump(dictStats, fileHandle, indent=2)
    fnPrintStats(dictStats)


def fnPrintStats(dictStats):
    """Print a side-by-side flux comparison."""
    for sKey, dictEntry in dictStats.items():
        print(f"{sKey:20s}: mean {dictEntry['mean']:.0f}, "
              f"95% CI [{dictEntry['ci95'][0]:.0f}, {dictEntry['ci95'][1]:.0f}]")


if __name__ == "__main__":
    main()
