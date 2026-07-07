#!/usr/bin/env python3
"""
Run vconverge for the Engle and Ribas cumulative XUV models.

Copies upstream prior files into each model subdirectory, backs up existing
converged output, runs vconverge, and restores the backup on failure.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

S_VCONVERGE_OUTPUT = "output/Converged_Param_Dictionary.json"
S_BACKUP_FILE = ".Converged_Param_Dictionary.json.bak"
S_VAIB_PREFIX = "[vaib]"


def fnBackupConvergedOutput(sModelDirectory, sFinalName):
    """Backup existing converged output if it exists."""
    sSource = os.path.join(sModelDirectory, sFinalName)
    sBackup = os.path.join(sModelDirectory, S_BACKUP_FILE)
    if os.path.exists(sSource):
        shutil.copy2(sSource, sBackup)


def fnRestoreConvergedOutput(sModelDirectory, sFinalName):
    """Restore converged output from backup if available."""
    sBackup = os.path.join(sModelDirectory, S_BACKUP_FILE)
    sTarget = os.path.join(sModelDirectory, sFinalName)
    if os.path.exists(sBackup):
        shutil.copy2(sBackup, sTarget)


def fnRemoveBackup(sModelDirectory):
    """Remove the backup file after a successful vconverge run."""
    sBackup = os.path.join(sModelDirectory, S_BACKUP_FILE)
    if os.path.exists(sBackup):
        os.remove(sBackup)


def fbRunVconverge(sModelDirectory):
    """Run vconverge in *sModelDirectory*, returning True on success."""
    print(f"{S_VAIB_PREFIX} {sModelDirectory}: running vconverge")
    result = subprocess.run(
        ["vconverge", "vconverge.in"],
        cwd=sModelDirectory
    )
    return result.returncode == 0


def fnRequireVconvergeOutput(sModelDirectory):
    """Verify vconverge produced its standard output file.

    The standard name (Converged_Param_Dictionary.json) is kept in
    place — output filenames are public API and the quantitative
    benchmarks bind to them; downstream steps reference the colliding
    basenames via qualified vaibify tokens.
    """
    sSource = os.path.join(sModelDirectory, S_VCONVERGE_OUTPUT)
    if not os.path.exists(sSource):
        raise FileNotFoundError(
            f"vconverge did not produce expected output: {sSource}"
        )


def fnCopyPriorFiles(sModelDirectory, dictPriorFiles):
    """Copy prior files from *dictPriorFiles* into *sModelDirectory*."""
    print(f"{S_VAIB_PREFIX} {sModelDirectory}: copying priors")
    for sSourcePath, sTargetName in dictPriorFiles.items():
        shutil.copy2(sSourcePath,
                     os.path.join(sModelDirectory, sTargetName))


def fbProcessModel(sModelDirectory, dictPriorFiles):
    """Copy priors, run vconverge, and handle backup for one model."""
    fnCopyPriorFiles(sModelDirectory, dictPriorFiles)
    fnBackupConvergedOutput(sModelDirectory, S_VCONVERGE_OUTPUT)
    bSuccess = fbRunVconverge(sModelDirectory)
    if bSuccess:
        fnRequireVconvergeOutput(sModelDirectory)
        fnRemoveBackup(sModelDirectory)
    else:
        print(f"{S_VAIB_PREFIX} {sModelDirectory}: vconverge failed, "
              "restoring backup")
        fnRestoreConvergedOutput(sModelDirectory, S_VCONVERGE_OUTPUT)
    return bSuccess


def fdictBuildModelConfiguration(sAgeSamples, sDynestySamples):
    """Build the mapping of model directories to their required priors."""
    return {
        "Engle": {sAgeSamples: "age_samples.txt"},
        "Ribas": {sDynestySamples: "dynesty_transform_final.npy"},
    }


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run vconverge for cumulative XUV models"
    )
    parser.add_argument("--age-samples", required=True,
                        help="Path to age samples (.txt)")
    parser.add_argument("--dynesty-samples", required=True,
                        help="Path to dynesty transform samples (.npy)")
    return parser.parse_args()


def main():
    """Run vconverge for all cumulative XUV models."""
    args = ftParseArguments()
    sScriptDirectory = str(Path(__file__).parent)
    os.chdir(sScriptDirectory)

    dictModels = fdictBuildModelConfiguration(
        args.age_samples, args.dynesty_samples
    )

    bAllSucceeded = True
    for sModelDirectory, dictPriorFiles in dictModels.items():
        bSuccess = fbProcessModel(sModelDirectory, dictPriorFiles)
        if not bSuccess:
            bAllSucceeded = False

    if not bAllSucceeded:
        print(f"{S_VAIB_PREFIX} One or more vconverge runs failed")
        sys.exit(1)
    print(f"{S_VAIB_PREFIX} All vconverge runs completed successfully")


if __name__ == "__main__":
    main()
