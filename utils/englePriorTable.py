"""
Generate vspace file-prior tables for the Engle XUV coefficients.

Each row is one joint draw of the four dXUVEngleMidLate coefficients in the
X-UV(5-1700 A) band, assembled from: (1) a row of the Stage-1 joint-refit
chain (native X-ray-band hinge coefficients with full covariance plus the
log-linear intrinsic-scatter law), (2) a slope/intercept draw from the
re-derived MUSCLES band conversion (2x2 covariance), (3) a persistent
per-realization deviate of the conversion's intrinsic scatter, and (4) an
optional per-realization offset z within the activity relation's population
scatter, evaluated at a reference age and folded into the intercept:

    a' = s a,   c' = s c,   d' = d,
    b' = s (b + z sigma_int(tau_ref)) + i + n_conv

vspace's predefined-prior mode draws all four columns from a common row, so
the correlations survive into the sweep. The constant-offset z treatment is
the bracket approximation to the age-dependent scatter law; the exact
treatment arrives with the vplanet scatter-law option.
"""

import json

import numpy as np


def fdictLoadConversion(sConversionFitPath):
    """Load the primary MUSCLES conversion block."""
    with open(sConversionFitPath) as fileHandle:
        dictFit = json.load(fileHandle)
    return dictFit["primary_fit_all_targets_rosat_band"]


def fdaDrawZ(sZMode, iRows, sZOffsetsPath):
    """Draw the per-realization scatter offset for the requested mode."""
    if sZMode == "none":
        return np.zeros(iRows)
    if sZMode == "population":
        return np.random.normal(0.0, 1.0, iRows)
    daZPosterior = np.loadtxt(sZOffsetsPath)
    return daZPosterior[np.random.randint(0, len(daZPosterior), iRows)]


def fdaSigmaIntAtReference(daRows, dictSummary, dTauReference):
    """Evaluate each row's native-band intrinsic scatter at the reference age."""
    return np.exp(daRows[:, 10] + daRows[:, 11]
                  * (dTauReference - dictSummary["dPivotTau"])
                  / dictSummary["dScaleTau"])


def fnWriteEnglePriorTable(sJointChainPath, sFitSummaryPath, sConversionFitPath,
                           sOutputPath, sZMode="none", sZOffsetsPath=None,
                           dTauReference=None, iRows=100000, iSeed=42):
    """Write a 4-column X-UV-band coefficient prior table for vspace."""
    np.random.seed(iSeed)
    daChain = np.load(sJointChainPath)
    with open(sFitSummaryPath) as fileHandle:
        dictSummary = json.load(fileHandle)
    dictConversion = fdictLoadConversion(sConversionFitPath)
    if dTauReference is None:
        dTauReference = dictSummary["dPivotTau"]
    daRows = daChain[np.random.randint(0, len(daChain), iRows)]
    daConversionDraws = np.random.multivariate_normal(
        [dictConversion["slope"], dictConversion["intercept"]],
        np.array(dictConversion["covariance_slope_intercept"]), iRows)
    daConversionScatter = np.random.normal(
        0.0, dictConversion["intrinsic_scatter_dex"]
        ["fScatterPosteriorMedian"], iRows)
    daZ = fdaDrawZ(sZMode, iRows, sZOffsetsPath)
    daSigmaInt = fdaSigmaIntAtReference(daRows, dictSummary, dTauReference)
    daSlope, daIntercept = daConversionDraws[:, 0], daConversionDraws[:, 1]
    daTable = np.column_stack([
        daSlope * daRows[:, 6],
        daSlope * (daRows[:, 7] + daZ * daSigmaInt)
        + daIntercept + daConversionScatter,
        daSlope * daRows[:, 8],
        daRows[:, 9]])
    np.savetxt(sOutputPath, daTable)
    print(f"[vaib] wrote {iRows}-row Engle X-UV prior table "
          f"(z-mode={sZMode}, tau_ref={dTauReference:.2f}) to {sOutputPath}")


def ftParseArguments():
    """Parse and return command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Write Engle X-UV coefficient prior tables for vspace.")
    parser.add_argument("--joint-chain", required=True)
    parser.add_argument("--fit-summary", required=True)
    parser.add_argument("--conversion-fit", required=True)
    parser.add_argument("--output", required=True, nargs="+",
                        help="One or more destination table paths.")
    parser.add_argument("--z-mode", default="none",
                        choices=["none", "population", "informed"])
    parser.add_argument("--z-offsets", default=None,
                        help="z posterior samples (required for informed).")
    parser.add_argument("--tau-reference", type=float, default=None,
                        help="log10(age/Gyr) at which sigma_int is evaluated.")
    return parser.parse_args()


def main():
    """Write one prior table per requested output path."""
    args = ftParseArguments()
    for sOutputPath in args.output:
        fnWriteEnglePriorTable(
            args.joint_chain, args.fit_summary, args.conversion_fit,
            sOutputPath, sZMode=args.z_mode, sZOffsetsPath=args.z_offsets,
            dTauReference=args.tau_reference)


if __name__ == "__main__":
    main()
