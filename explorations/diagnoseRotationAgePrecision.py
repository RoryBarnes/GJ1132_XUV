#!/usr/bin/env python3
"""
Diagnose why the joint refit gives GJ 1132 a +/-0.026 dex rotation age.

Part 1 decomposes the A11 log-age spread by ablation: each source (P_rot
measurement, coefficient covariance, intrinsic scatter) is frozen in turn and
the drop in variance is attributed to it (first-order indices; Saltelli et al.
2008, Global Sensitivity Analysis: The Primer).

Part 2 asks how much rotation-age intrinsic scatter the benchmark stars allow
once their latent true ages are integrated out, rather than sampled jointly as
the refit does. For a constant scatter sigma it computes the benchmark
marginal likelihood, integrating each star's latent log-age by Gauss-Hermite
quadrature against its split-normal age error, and profiles over the four
hinge coefficients. Sampling latents jointly with their scale is the
"centered" hierarchical parameterization whose funnel geometry can trap
samplers at small scale (Neal 2003, Ann. Stat. 31, 705; Betancourt & Girolami
2015, in Current Trends in Bayesian Methodology, arXiv:1312.0906). The profile
uses the same benchmark sample, filters, and split-normal errors as the refit
but omits the X-ray terms, which constrain rotation scatter only weakly.

Usage: python diagnoseRotationAgePrecision.py --joint-chain <npy>
       --fit-summary <json> --joint-sample <json> --output <json>
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import optimize

S_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S_REPOSITORY)
sys.path.insert(0, os.path.join(S_REPOSITORY, "EngleAgeDistribution"))
from utils import dataRefitJointRelations as refit  # noqa: E402
import dataAge  # noqa: E402

I_SEED = 42
I_DRAWS = 100000
DA_SCATTER_GRID = np.logspace(-3, np.log10(0.5), 30)


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for sName in ("joint-chain", "fit-summary", "joint-sample", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdictDrawAgeSources(daChain, dictSummary):
    """Draw every source of the A11 log-age once, as dataAge.py combines them."""
    daRows = dataAge.fdaDrawChainRows(daChain, I_DRAWS)
    daProt = dataAge.fdaSampleSplitNormal(dataAge.D_PROT_MEAN, dataAge.D_PROT_PLUS,
                                          dataAge.D_PROT_MINUS, I_DRAWS)
    return {"daRows": daRows, "daProt": daProt,
            "daDeviate": np.random.normal(0.0, 1.0, I_DRAWS),
            "dictSummary": dictSummary}


def fdaComposeLogAge(dictSources, bFreezeProt=False, bFreezeRows=False,
                     bZeroScatter=False):
    """Compose log-ages, optionally freezing sources at their means."""
    daRows, daProt = dictSources["daRows"], dictSources["daProt"]
    if bFreezeRows:
        daRows = np.tile(np.mean(daRows, axis=0), (I_DRAWS, 1))
    if bFreezeProt:
        daProt = np.full(I_DRAWS, dataAge.D_PROT_MEAN)
    daSigma = dataAge.fdaEvaluateScatterLaw(daRows, daProt, dictSources["dictSummary"])
    daScatter = 0.0 if bZeroScatter else dictSources["daDeviate"] * daSigma
    return dataAge.fdaEvaluateHinge(daRows[:, 0:4], daProt) + daScatter


def fdictDecomposeAgeSpread(daChain, dictSummary):
    """Part 1: first-order variance shares of the A11 log-age."""
    dictSources = fdictDrawAgeSources(daChain, dictSummary)
    dTotal = float(np.var(fdaComposeLogAge(dictSources)))
    dictAblations = {"rotation period measurement": {"bFreezeProt": True},
                     "coefficient covariance": {"bFreezeRows": True},
                     "intrinsic scatter": {"bZeroScatter": True}}
    dictShares = {}
    for sName, dictFreeze in dictAblations.items():
        dRemaining = float(np.var(fdaComposeLogAge(dictSources, **dictFreeze)))
        dictShares[sName] = {"dVarianceShare": (dTotal - dRemaining) / dTotal,
                             "dSpreadDex": float(np.sqrt(max(dTotal - dRemaining, 0)))}
    daSigma = dataAge.fdaEvaluateScatterLaw(dictSources["daRows"],
                                            dictSources["daProt"], dictSummary)
    return {"dTotalSpreadDex": float(np.sqrt(dTotal)), "dictSourceShares": dictShares,
            "daScatterAtProtPercentiles": [float(x) for x in
                                           np.percentile(daSigma, [2.5, 50, 97.5])]}


def fdBenchmarkMarginalLogLike(daCoefficients, dScatter, dictBench):
    """Benchmark log-likelihood with each latent log-age integrated out."""
    daSlope = refit.fdaLocalSlope(daCoefficients, dictBench["daProt"])
    daSpread = np.sqrt(dScatter ** 2 + (daSlope * dictBench["daSigProt"]) ** 2)
    daMean = refit.fdaHinge(daCoefficients, dictBench["daProt"])
    daLatent = daMean[:, None] + daSpread[:, None] * refit.DA_NODES[None, :]
    daLogPdf = refit.fdaSplitNormalLogPdf(
        dictBench["daTauObs"][:, None], daLatent,
        dictBench["daSigPlus"][:, None], dictBench["daSigMinus"][:, None])
    daTerms = refit.DA_LOG_WEIGHTS[None, :] + daLogPdf
    daMax = np.max(daTerms, axis=1)
    return float(np.sum(daMax + np.log(np.sum(np.exp(daTerms - daMax[:, None]), axis=1))))


def fdProfileLogLike(dScatter, dictBench, daStart):
    """Maximize the benchmark marginal likelihood over the hinge at one scatter."""
    def fdNegative(daCoefficients):
        return -fdBenchmarkMarginalLogLike(daCoefficients, dScatter, dictBench)
    result = optimize.minimize(fdNegative, daStart, method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    return -float(result.fun)


def fbHasPopulationAge(dictRecord):
    """Return whether a benchmark's age is a shared population assignment."""
    return "Population" in dictRecord["dictAgeConstraint"].get("sMethod", "")


def fdictBuildBenchmarks(sJointSample, bExcludePopulationAges):
    """Build the refit's mid-late benchmark block, optionally without population ages."""
    with open(sJointSample) as fileHandle:
        listRecords = json.load(fileHandle)
    listKept = refit.flistFilterRecords(listRecords, "midlate", True)
    dictSplit = refit.fdictAssembleArrays(listKept)
    listBench = [d for d in dictSplit["listBench"]
                 if not (bExcludePopulationAges and fbHasPopulationAge(d))]
    return refit.fdictBenchmarkBlock(listBench, dictSplit["dMedianRelative"])


def fdictProfileScatter(sJointSample, daStart, bExcludePopulationAges):
    """Part 2: profile log-likelihood of a constant rotation-age scatter."""
    dictBench = fdictBuildBenchmarks(sJointSample, bExcludePopulationAges)
    daProfile = np.array([fdProfileLogLike(dScatter, dictBench, daStart)
                          for dScatter in DA_SCATTER_GRID])
    daDelta = daProfile - np.max(daProfile)
    baWithinTwo = daDelta > -2.0
    return {"iBenchmarks": int(len(dictBench["daProt"])),
            "daScatterDex": DA_SCATTER_GRID.tolist(), "daDeltaLogLike": daDelta.tolist(),
            "dScatterAtMaximumDex": float(DA_SCATTER_GRID[np.argmax(daProfile)]),
            "dLargestScatterWithinTwoLogLikeDex": float(DA_SCATTER_GRID[baWithinTwo][-1]),
            "dBestFitAgeAtGj1132ProtGyr": fdBestFitAge(
                dictBench, daStart, float(DA_SCATTER_GRID[np.argmax(daProfile)]))}


def fdBestFitAge(dictBench, daStart, dScatter):
    """Return the maximum-likelihood hinge age (Gyr) at GJ 1132's period."""
    result = optimize.minimize(
        lambda daC: -fdBenchmarkMarginalLogLike(daC, dScatter, dictBench), daStart,
        method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    return float(10 ** refit.fdaHinge(result.x, np.array([dataAge.D_PROT_MEAN]))[0])


def fdictSummarizeBenchmarkResiduals(sJointSample, daCoefficients):
    """Part 3: benchmark log-age residuals about the hinge in units of their errors."""
    dictBench = fdictBuildBenchmarks(sJointSample, False)
    daResidual = dictBench["daTauObs"] - refit.fdaHinge(daCoefficients, dictBench["daProt"])
    daSigma = np.where(daResidual < 0, dictBench["daSigPlus"], dictBench["daSigMinus"])
    daSigmaSlope = refit.fdaLocalSlope(daCoefficients, dictBench["daProt"]) * dictBench["daSigProt"]
    daNormalized = daResidual / np.sqrt(daSigma ** 2 + daSigmaSlope ** 2)
    baOldTrack = dictBench["daProt"] >= daCoefficients[3]
    return {"daProtDays": dictBench["daProt"].tolist(),
            "daResidualDex": daResidual.tolist(), "daNormalizedResidual": daNormalized.tolist(),
            "dMeanSquareNormalizedAll": float(np.mean(daNormalized ** 2)),
            "dMeanSquareNormalizedOldTrack": float(np.mean(daNormalized[baOldTrack] ** 2)),
            "dRmsResidualOldTrackDex": float(np.sqrt(np.mean(daResidual[baOldTrack] ** 2))),
            "dMedianAgeErrorOldTrackDex": float(np.median(daSigma[baOldTrack])),
            "iOldTrack": int(np.sum(baOldTrack))}


def main():
    """Write every diagnostic to one JSON file."""
    np.random.seed(I_SEED)
    args = ftParseArguments()
    daChain = np.load(args.joint_chain)
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    daStart = np.mean(daChain[:, 0:4], axis=0)
    dictReport = {"dictAgeSpread": fdictDecomposeAgeSpread(daChain, dictSummary),
                  "dictScatterProfile": fdictProfileScatter(
                      args.joint_sample, daStart, False),
                  "dictScatterProfileWithoutPopulationAges": fdictProfileScatter(
                      args.joint_sample, daStart, True),
                  "dictBenchmarkResiduals": fdictSummarizeBenchmarkResiduals(
                      args.joint_sample, daStart),
                  "daChainLogScatterInterceptPercentiles": [
                      float(x) for x in np.percentile(daChain[:, 4], [2.5, 50, 97.5])],
                  "dScatterFloorDex": refit.D_MIN_SCATTER}
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
