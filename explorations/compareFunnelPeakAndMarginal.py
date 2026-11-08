#!/usr/bin/env python3
"""
Quantify the funnel in the A08 refit's centered parameterization.

For a constant rotation-age scatter sigma (other hyperparameters fixed at the
marginalized chain's mean), compare two benchmark log-densities: (1) the PEAK
of the joint density over the 34 latent true log-ages, which is what an MCMC
sampler of the centered model is drawn toward, and (2) the MARGINAL density,
which integrates the latents out and is what the posterior of sigma actually
depends on. Their difference is the log-volume of latent space compatible
with sigma; a peak that grows without bound while the marginal falls is the
funnel of Neal (2003, Ann. Stat. 31, 705); see also Papaspiliopoulos, Roberts
& Skold (2007, Stat. Sci. 22, 59) on centered versus non-centered
parameterizations.

Usage: python compareFunnelPeakAndMarginal.py --joint-sample <json>
       --marginal-chain <npy> --output <json>
"""

import argparse
import json
import os
import sys

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S_DIRECTORY)
import refitMarginalizedLatentAges as marginal  # noqa: E402
from utils import dataRefitJointRelations as refit  # noqa: E402

DA_SCATTER_DEX = np.array([0.001, 0.002, 0.003, 0.01, 0.025, 0.05, 0.1])


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for sName in ("joint-sample", "marginal-chain", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdLogPeakAndMarginal(daTheta, dictData):
    """Return the benchmark log-density maximized and integrated over latents."""
    tLatentPrior = marginal.ftLatentPrior(daTheta, dictData)
    dPeak, dMarginal = 0.0, 0.0
    for iStar in range(dictData["iBenchmarks"]):
        dCenter, dWidth = marginal.ftProposalForStar(iStar, dictData["dictBench"], tLatentPrior)
        daTau = dCenter + dWidth * np.linspace(-12.0, 12.0, 24001)
        dPeak += float(np.max(marginal.fdaLogIntegrand(daTau, iStar, daTheta, dictData,
                                                       tLatentPrior)))
        dMarginal += marginal.fdLogStarIntegral(iStar, daTheta, dictData, tLatentPrior)
    return dPeak, dMarginal


def flistCompareAcrossScatter(daTheta, dictData):
    """Tabulate peak and marginal log-densities on a scatter grid."""
    listRows = []
    for dScatter in DA_SCATTER_DEX:
        daTrial = daTheta.copy()
        daTrial[4], daTrial[5] = np.log(dScatter), 0.0
        dPeak, dMarginal = fdLogPeakAndMarginal(daTrial, dictData)
        listRows.append({"dScatterDex": float(dScatter), "dLogPeak": dPeak,
                         "dLogMarginal": dMarginal, "dLogVolume": dMarginal - dPeak})
    return listRows


def main():
    """Write the peak-versus-marginal table to a JSON file."""
    args = ftParseArguments()
    dictData = refit.fdictBuildData(args.joint_sample, "midlate", True)
    daTheta = np.mean(np.load(args.marginal_chain), axis=0)
    listRows = flistCompareAcrossScatter(daTheta, dictData)
    with open(args.output, "w") as fileHandle:
        json.dump({"iLatentAges": dictData["iBenchmarks"], "listRows": listRows},
                  fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
