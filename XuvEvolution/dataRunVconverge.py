#!/usr/bin/env python3
"""
Run vconverge for the EngleBarnes and RibasBarnes XUV evolution models.

Copies upstream prior files into each model subdirectory, backs up existing
converged output, runs vconverge, and restores the backup on failure.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

S_CONVERGED_FILE = "output/Converged_Param_Dictionary.json"
S_BACKUP_FILE = ".Converged_Param_Dictionary.json.bak"
SA_MODEL_DIRECTORIES = ["EngleBarnes", "RibasBarnes"]
S_VAIB_PREFIX = "[vaib]"


def fnCopyUpstreamPriors(sModelDirectory, sFlareSource, sAgeSource,
                         sDynestySource):
    """Copy flare, age, and dynesty prior files into *sModelDirectory*."""
    print(f"{S_VAIB_PREFIX} {sModelDirectory}: copying priors")
    shutil.copy2(sFlareSource,
                 os.path.join(sModelDirectory, "flares_variable_slope.npy"))
    shutil.copy2(sAgeSource,
                 os.path.join(sModelDirectory, "age_samples.txt"))
    shutil.copy2(sDynestySource,
                 os.path.join(sModelDirectory, "dynesty_transform_final.npy"))


def fnBackupConvergedOutput(sModelDirectory):
    """Backup existing converged output if it exists."""
    sSource = os.path.join(sModelDirectory, S_CONVERGED_FILE)
    sBackup = os.path.join(sModelDirectory, S_BACKUP_FILE)
    if os.path.exists(sSource):
        shutil.copy2(sSource, sBackup)


def fnRestoreConvergedOutput(sModelDirectory):
    """Restore converged output from backup if available."""
    sBackup = os.path.join(sModelDirectory, S_BACKUP_FILE)
    sTarget = os.path.join(sModelDirectory, S_CONVERGED_FILE)
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


def fbProcessModel(sModelDirectory, sFlareSource, sAgeSource,
                   sDynestySource):
    """Copy priors, run vconverge, and handle backup for one model."""
    fnCopyUpstreamPriors(sModelDirectory, sFlareSource, sAgeSource,
                         sDynestySource)
    fnBackupConvergedOutput(sModelDirectory)
    bSuccess = fbRunVconverge(sModelDirectory)
    if bSuccess:
        fnRemoveBackup(sModelDirectory)
    else:
        print(f"{S_VAIB_PREFIX} {sModelDirectory}: vconverge failed, "
              "restoring backup")
        fnRestoreConvergedOutput(sModelDirectory)
    return bSuccess


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run vconverge for XUV evolution models"
    )
    parser.add_argument("--flare-samples", required=True,
                        help="Path to flare MCMC samples (.npy)")
    parser.add_argument("--age-samples", required=True,
                        help="Path to age samples (.txt)")
    parser.add_argument("--dynesty-samples", required=True,
                        help="Path to dynesty transform samples (.npy)")
    return parser.parse_args()


def main():
    """Run vconverge for all XUV evolution models."""
    args = ftParseArguments()
    sScriptDirectory = str(Path(__file__).parent)
    os.chdir(sScriptDirectory)

    bAllSucceeded = True
    for sModelDirectory in SA_MODEL_DIRECTORIES:
        bSuccess = fbProcessModel(
            sModelDirectory,
            args.flare_samples,
            args.age_samples,
            args.dynesty_samples,
        )
        if not bSuccess:
            bAllSucceeded = False

    if not bAllSucceeded:
        print(f"{S_VAIB_PREFIX} One or more vconverge runs failed")
        sys.exit(1)
    print(f"{S_VAIB_PREFIX} All vconverge runs completed successfully")


if __name__ == "__main__":
    main()
