#!/usr/bin/env python3
"""
Package the joint-refit posteriors into one catalog-ready coefficient file.

For each fit variant (two M dwarf sub-classes x with/without halo subdwarfs)
this records the posterior mean vector and full 12x12 covariance of the
hyperparameters (rotation hinge a,b,c,d + scatter law e,f; activity hinge
a,b,c,d + scatter law e,f), the scatter-law pivots needed to evaluate
sigma_int, and the convergence diagnostics. The re-derived MUSCLES band
conversion (slope/intercept covariance + intrinsic scatter) rides along so a
catalog consumer has every number needed to predict an individual M dwarf's
X-UV history with honest uncertainties from this single file.

Canonical variants (per project decision 2026-07-14) are the *_withsd fits.
"""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SA_TAGS = ["midlate_withsd", "midlate_nosd", "early_withsd", "early_nosd"]
SA_PARAMETERS = ["a_rot", "b_rot", "c_rot", "d_rot", "e_rot", "f_rot",
                 "a_act", "b_act", "c_act", "d_act", "e_act", "f_act"]


def fdictPackageOneFit(sTag):
    """Build the posterior block for one fit variant."""
    daChain = np.load(os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy"))
    with open(os.path.join(S_DIRECTORY,
                           f"jointFitSummary_{sTag}.json")) as fileHandle:
        dictSummary = json.load(fileHandle)
    return {
        "bCanonical": sTag.endswith("_withsd"),
        "saParameterNames": SA_PARAMETERS,
        "daPosteriorMean": [float(dV) for dV in np.mean(daChain, axis=0)],
        "daPosteriorCovariance": np.cov(daChain, rowvar=False).tolist(),
        "dictScatterLawPivots": {
            "dPivotProt": dictSummary["dPivotProt"],
            "dScaleProt": dictSummary["dScaleProt"],
            "dPivotTau": dictSummary["dPivotTau"],
            "dScaleTau": dictSummary["dScaleTau"]},
        "dictConvergence": {"bConverged": dictSummary["bConverged"],
                            "iProposals": dictSummary["iSteps"],
                            "dTauMax": dictSummary["dTauMax"],
                            "iPosteriorDraws": dictSummary["iPosteriorDraws"]},
    }


def fdictPackageConversion():
    """Extract the MUSCLES conversion block for catalog consumers."""
    with open(os.path.join(S_DIRECTORY, "musclesConversion",
                           "conversionFit.json")) as fileHandle:
        dictFit = json.load(fileHandle)
    dictPrimary = dictFit["primary_fit_all_targets_rosat_band"]
    return {
        "sRelation": dictFit["relation"],
        "dSlope": dictPrimary["slope"],
        "dIntercept": dictPrimary["intercept"],
        "daCovariance": dictPrimary["covariance_slope_intercept"],
        "dictIntrinsicScatterDex": dictPrimary["intrinsic_scatter_dex"],
        "iStars": dictPrimary["n_stars"],
    }


def main():
    """Assemble and save the catalog-ready posterior package."""
    dictPackage = {
        "sDescription": "Joint hierarchical refit of the Engle & Guinan "
                        "(2023) and Engle (2024) M dwarf relations, with the "
                        "re-derived MUSCLES X-UV band conversion.",
        "sValidity": "M0-6.5 dwarfs only; no calibrated relation later than "
                     "M6.5.",
        "dictConversionXuv": fdictPackageConversion(),
    }
    for sTag in SA_TAGS:
        dictPackage[sTag] = fdictPackageOneFit(sTag)
    sPath = os.path.join(S_DIRECTORY, "engleCoefficientPosteriors.json")
    with open(sPath, "w") as fileHandle:
        json.dump(dictPackage, fileHandle, indent=2)
    print(f"Saved engleCoefficientPosteriors.json "
          f"({len(SA_TAGS)} fits + conversion)")


if __name__ == "__main__":
    main()
