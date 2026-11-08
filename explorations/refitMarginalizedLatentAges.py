#!/usr/bin/env python3
"""
Refit the mid-late joint model with benchmark latent ages integrated out.

The A08 refit samples one latent true log-age per benchmark star jointly with
the hyperparameters (the centered hierarchical parameterization). That
geometry can trap samplers at small intrinsic scatter (Neal 2003, Ann. Stat.
31, 705; Betancourt & Girolami 2015, arXiv:1312.0906). This script keeps the
model identical - same sample, priors, scatter floor, field-star and ensemble
terms, imported from utils/dataRefitJointRelations.py - but integrates each
benchmark's latent age numerically, so emcee samples only the 12
hyperparameters. Each integral uses Gauss-Hermite nodes centered on a
Gaussian approximation to that star's latent-age posterior (importance
quadrature), checked against a brute-force grid before sampling. Agreement
with the A08 chain means the sampler was not trapped; disagreement means it
was.

Usage: python refitMarginalizedLatentAges.py --joint-sample <json>
       --joint-chain <npy> --fit-summary <json> --output <json>
"""

import argparse
import json
import os
import sys

import numpy as np
import emcee

S_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S_REPOSITORY)
sys.path.insert(0, os.path.join(S_REPOSITORY, "EngleAgeDistribution"))
from utils import dataRefitJointRelations as refit  # noqa: E402
import dataAge  # noqa: E402

I_SEED = 42
I_WALKERS = 64
I_CHUNK = 2000
I_MAX_STEPS = 60000
D_PROPOSAL_WIDENING = 1.5
DA_GRID = np.linspace(-2.0, 1.3, 200001)


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for sName in ("joint-sample", "joint-chain", "fit-summary", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdaLogIntegrand(daTau, iStar, daTheta, dictData, tLatentPrior):
    """Log of latent prior x age likelihood x optional X-ray likelihood."""
    dictBench = dictData["dictBench"]
    daMu, daSpread = tLatentPrior
    daLog = (-0.5 * ((daTau - daMu[iStar]) / daSpread[iStar]) ** 2
             - np.log(daSpread[iStar] * np.sqrt(2.0 * np.pi)))
    daLog = daLog + refit.fdaSplitNormalLogPdf(
        dictBench["daTauObs"][iStar], daTau,
        dictBench["daSigPlus"][iStar], dictBench["daSigMinus"][iStar])
    if dictBench["baHasXray"][iStar]:
        daScatter = refit.fdaScatterLaw(daTheta[10], daTheta[11], daTau,
                                        dictData["dPivotTau"], dictData["dScaleTau"])
        daVariance = dictBench["daYErr"][iStar] ** 2 + daScatter ** 2
        daLog = daLog - 0.5 * ((dictBench["daY"][iStar] - refit.fdaHinge(
            daTheta[6:10], daTau)) ** 2 / daVariance + np.log(2 * np.pi * daVariance))
    return daLog


def ftProposalForStar(iStar, dictBench, tLatentPrior):
    """Gaussian approximation to one star's latent-age posterior."""
    daMu, daSpread = tLatentPrior
    dSigmaObserved = 0.5 * (dictBench["daSigPlus"][iStar] + dictBench["daSigMinus"][iStar])
    dVariance = 1.0 / (1.0 / daSpread[iStar] ** 2 + 1.0 / dSigmaObserved ** 2)
    dCenter = dVariance * (daMu[iStar] / daSpread[iStar] ** 2
                           + dictBench["daTauObs"][iStar] / dSigmaObserved ** 2)
    return dCenter, D_PROPOSAL_WIDENING * np.sqrt(dVariance)


def fdLogStarIntegral(iStar, daTheta, dictData, tLatentPrior):
    """Importance Gauss-Hermite integral of one star's latent log-age."""
    dCenter, dWidth = ftProposalForStar(iStar, dictData["dictBench"], tLatentPrior)
    daTau = dCenter + dWidth * refit.DA_NODES
    daTerms = (np.log(refit.DA_WEIGHTS) + 0.5 * refit.DA_NODES ** 2
               + fdaLogIntegrand(daTau, iStar, daTheta, dictData, tLatentPrior))
    dMax = np.max(daTerms)
    return float(np.log(dWidth) + dMax + np.log(np.sum(np.exp(daTerms - dMax))))


def ftLatentPrior(daTheta, dictData):
    """Return the latent-age prior mean and spread for every benchmark."""
    dictBench = dictData["dictBench"]
    daSpread = refit.fdaLatentAgeSpread(daTheta[0:4], daTheta[4], daTheta[5],
                                        dictBench["daProt"], dictBench["daSigProt"],
                                        dictData)
    return refit.fdaHinge(daTheta[0:4], dictBench["daProt"]), daSpread


def fdBenchmarkMarginalLogLike(daTheta, dictData):
    """Sum of every benchmark's integrated (rotation + X-ray) likelihood."""
    tLatentPrior = ftLatentPrior(daTheta, dictData)
    if tLatentPrior[1] is None:
        return -np.inf
    return sum(fdLogStarIntegral(iStar, daTheta, dictData, tLatentPrior)
               for iStar in range(dictData["iBenchmarks"]))


def fdNonBenchmarkLogLike(daTheta, dictData):
    """Field-star and cluster-ensemble terms, exactly as in the A08 refit."""
    dictField, dictEnsemble = dictData["dictField"], dictData["dictEnsemble"]
    daSpread = refit.fdaLatentAgeSpread(daTheta[0:4], daTheta[4], daTheta[5],
                                        dictField["daProt"], dictField["daSigProt"], dictData)
    if daSpread is None:
        return -np.inf
    daField = refit.fdaQuadratureActivityLogLike(
        refit.fdaHinge(daTheta[0:4], dictField["daProt"]), daSpread, dictField["daY"],
        dictField["daYErr"], daTheta[6:10], daTheta[10], daTheta[11], dictData)
    daEnsemble = refit.fdaQuadratureActivityLogLike(
        dictEnsemble["daTau"], dictEnsemble["daSigTau"], dictEnsemble["daY"],
        dictEnsemble["daYErr"], daTheta[6:10], daTheta[10], daTheta[11], dictData)
    if daField is None or daEnsemble is None:
        return -np.inf
    return float(np.sum(daField) + np.sum(daEnsemble))


def fdMarginalLogPosterior(daTheta, dictData, dictBounds):
    """Log-posterior over the 12 hyperparameters with latents integrated out."""
    dLogPrior = refit.fdHyperLogPrior(daTheta, dictBounds)
    if not np.isfinite(dLogPrior):
        return -np.inf
    dBenchmark = fdBenchmarkMarginalLogLike(daTheta, dictData)
    if not np.isfinite(dBenchmark):
        return -np.inf
    return dLogPrior + dBenchmark + fdNonBenchmarkLogLike(daTheta, dictData)


def fdMaximumQuadratureError(daTheta, dictData):
    """Largest |quadrature - brute-force grid| log-integral over all benchmarks."""
    tLatentPrior = ftLatentPrior(daTheta, dictData)
    dWorst = 0.0
    for iStar in range(dictData["iBenchmarks"]):
        daLog = fdaLogIntegrand(DA_GRID, iStar, daTheta, dictData, tLatentPrior)
        dMax = np.max(daLog)
        dGrid = dMax + np.log(np.trapz(np.exp(daLog - dMax), DA_GRID))
        dWorst = max(dWorst, abs(dGrid - fdLogStarIntegral(iStar, daTheta, dictData,
                                                            tLatentPrior)))
    return float(dWorst)


def fdaRunSampler(dictData, daStart):
    """Run emcee with the A08 moves until 50 autocorrelation times or the cap."""
    np.random.seed(I_SEED)
    daInitial = daStart + 1e-4 * np.random.randn(I_WALKERS, len(daStart))
    sampler = emcee.EnsembleSampler(
        I_WALKERS, len(daStart), fdMarginalLogPosterior,
        args=(dictData, refit.DICT_PUBLISHED["midlate"]), moves=refit.LIST_MOVES)
    state, dTau = daInitial, np.inf
    while sampler.iteration < I_MAX_STEPS and sampler.iteration < 50 * dTau:
        state = sampler.run_mcmc(state, I_CHUNK, progress=False)
        dTau = float(np.max(sampler.get_autocorr_time(tol=0)))
        print(f"  steps={sampler.iteration} tau={dTau:.0f}", flush=True)
    iDiscard = int(min(5 * dTau, sampler.iteration // 2))
    daFlat = sampler.get_chain(discard=iDiscard, thin=max(1, int(dTau / 2)), flat=True)
    return daFlat, sampler.iteration, dTau


def fdictSummarizeScatter(daChain, dictSummary):
    """Rotation scatter at GJ 1132's period and the implied A11 log-age."""
    np.random.seed(I_SEED)
    daSigma = dataAge.fdaEvaluateScatterLaw(
        daChain, np.full(len(daChain), dataAge.D_PROT_MEAN), dictSummary)
    daLogAge = dataAge.fdaComputeRotationOnlyAges(daChain, dictSummary)
    daAge = 10 ** daLogAge[daLogAge <= dataAge.D_MAX_LOG_AGE]
    return {"daScatterAtGj1132Percentiles": np.percentile(daSigma, [2.5, 16, 50, 84, 97.5]).tolist(),
            "daERotPercentiles": np.percentile(daChain[:, 4], [2.5, 50, 97.5]).tolist(),
            "daFRotPercentiles": np.percentile(daChain[:, 5], [2.5, 50, 97.5]).tolist(),
            "dAgeMedianGyr": float(np.median(daAge)),
            "daAgeCi95Gyr": np.percentile(daAge, [2.5, 97.5]).tolist(),
            "dLogAgeSpreadDex": float(np.std(daLogAge))}


def main():
    """Sample the marginalized model and compare with the A08 chain."""
    args = ftParseArguments()
    dictData = refit.fdictBuildData(args.joint_sample, "midlate", True)
    daChain = np.load(args.joint_chain)
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    daStart = np.mean(daChain, axis=0)
    dQuadratureError = fdMaximumQuadratureError(daStart, dictData)
    print(f"max quadrature error at the A08 mean: {dQuadratureError:.2e} nats", flush=True)
    daMarginal, iSteps, dTau = fdaRunSampler(dictData, daStart)
    dictReport = {"dMaxQuadratureErrorNats": dQuadratureError, "iSteps": int(iSteps),
                  "dAutocorrelationSteps": dTau, "iDraws": int(len(daMarginal)),
                  "dictA08Chain": fdictSummarizeScatter(daChain, dictSummary),
                  "dictMarginalized": fdictSummarizeScatter(daMarginal, dictSummary)}
    np.save(os.path.splitext(args.output)[0] + "Chain.npy", daMarginal)
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
