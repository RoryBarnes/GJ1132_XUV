#!/usr/bin/env python3
"""
Run independent emcee runs of the A08 mid-late model from dispersed starts.

--model centered samples A08's own 46-dimensional posterior (12
hyperparameters plus 34 benchmark latent log-ages, utils/dataRefitJointRelations
.fdJointLogPosterior). --model marginalized samples the same model with the
latent ages marginalized (explorations/refitMarginalizedLatentAges.py). Every
walker of every run starts from an independent draw over a wide box - the
rotation scatter alone is log-uniform from 0.003 to 0.3 dex - instead of A08's
1e-4 ball, and the runs use A08's moves. Agreement across runs is measured
with split-R-hat (Gelman et al. 2013, Bayesian Data Analysis, 3rd ed.,
Section 11.4); each walker's rotation-scatter trace is saved so trapping can
be seen directly.

Usage: python compareSamplersMultiRun.py --model centered --runs 4
       --steps 84000 --walkers 128 --thin 20 --joint-sample <json>
       --fit-summary <json> --output <json>
"""

import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import emcee

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S_DIRECTORY)
import refitMarginalizedLatentAges as marginal  # noqa: E402
from utils import dataRefitJointRelations as refit  # noqa: E402

I_HYPER = refit.I_HYPER_DIMENSIONS
I_BASE_SEED = 1000
TA_SCATTER_START_DEX = (0.003, 0.3)


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=["centered", "marginalized"])
    for sName in ("runs", "steps", "walkers", "thin"):
        parser.add_argument(f"--{sName}", required=True, type=int)
    for sName in ("joint-sample", "fit-summary", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdaDrawHyperStart(generator):
    """Draw one hyperparameter start from a wide box around the published fit."""
    dictPub = refit.DICT_PUBLISHED["midlate"]
    daRotation = np.array(dictPub["daRotation"])
    daActivity = np.array(dictPub["daActivityRatio"])
    return np.concatenate([
        daRotation[:3] * generator.uniform(0.6, 1.4, 3),
        [generator.uniform(15.0, 40.0),
         generator.uniform(*np.log(TA_SCATTER_START_DEX)),
         generator.uniform(-1.0, 1.0)],
        daActivity[:3] + generator.normal(0.0, [0.1, 0.2, 0.5]),
        [generator.uniform(-0.5, 0.8), np.log(generator.uniform(0.1, 1.0)),
         generator.uniform(-1.0, 1.0)]])


def fdaDrawOneStart(generator, dictData, bCentered):
    """Draw one walker start (plus latent ages for the centered model)."""
    daStart = fdaDrawHyperStart(generator)
    if not bCentered:
        return daStart
    dictBench = dictData["dictBench"]
    daLatent = dictBench["daTauObs"] + generator.normal(
        0.0, 0.5 * (dictBench["daSigPlus"] + dictBench["daSigMinus"]))
    return np.concatenate([daStart, daLatent])


def fdLogPosterior(daTheta, dictData, bCentered):
    """Dispatch to the centered (A08) or marginalized log-posterior."""
    dictBounds = refit.DICT_PUBLISHED["midlate"]
    if bCentered:
        return refit.fdJointLogPosterior(daTheta, dictData, dictBounds)
    return marginal.fdMarginalLogPosterior(daTheta, dictData, dictBounds)


def fdaDrawStarts(iSeed, iWalkers, dictData, bCentered):
    """Draw dispersed starts, redrawing any walker whose posterior is not finite."""
    generator = np.random.default_rng(iSeed)
    listStarts = []
    while len(listStarts) < iWalkers:
        daCandidate = fdaDrawOneStart(generator, dictData, bCentered)
        if np.isfinite(fdLogPosterior(daCandidate, dictData, bCentered)):
            listStarts.append(daCandidate)
    return np.array(listStarts)


def fdictRunOne(tTask):
    """Run one independent emcee run; return its post-burn draws and traces."""
    iRun, sModel, iSteps, iWalkers, iThin, sJointSample = tTask
    bCentered = sModel == "centered"
    np.random.seed(I_BASE_SEED + iRun)
    dictData = refit.fdictBuildData(sJointSample, "midlate", True)
    daStart = fdaDrawStarts(I_BASE_SEED + iRun, iWalkers, dictData, bCentered)
    sampler = emcee.EnsembleSampler(iWalkers, daStart.shape[1], fdLogPosterior,
                                    args=(dictData, bCentered), moves=refit.LIST_MOVES)
    sampler.run_mcmc(daStart, iSteps // iThin, thin_by=iThin, progress=False)
    daChain = sampler.get_chain()
    iBurn = daChain.shape[0] // 2
    print(f"run {iRun} done: acceptance {np.mean(sampler.acceptance_fraction):.3f}",
          flush=True)
    return {"daPostBurn": daChain[iBurn:, :, :I_HYPER],
            "daScatterTrace": daChain[:, :, 4],
            "daTau": sampler.get_autocorr_time(tol=0)[:I_HYPER] * iThin,
            "dAcceptance": float(np.mean(sampler.acceptance_fraction))}


def fdaSplitRhat(listPostBurn):
    """Split-R-hat per hyperparameter, treating each run's draws as one chain."""
    listChains = []
    for daRun in listPostBurn:
        daFlat = daRun.reshape(-1, daRun.shape[-1], order="F")
        iHalf = len(daFlat) // 2
        listChains.extend([daFlat[:iHalf], daFlat[iHalf:2 * iHalf]])
    daChains = np.array(listChains)
    iLength = daChains.shape[1]
    daWithin = np.mean(np.var(daChains, axis=1, ddof=1), axis=0)
    daBetween = iLength * np.var(np.mean(daChains, axis=1), axis=0, ddof=1)
    daPooled = (iLength - 1) / iLength * daWithin + daBetween / iLength
    return np.sqrt(daPooled / daWithin)


def fdictSummarizeRuns(listResults, dictSummary):
    """Per-run and pooled scatter and age summaries, plus split-R-hat."""
    listPostBurn = [d["daPostBurn"] for d in listResults]
    daPooled = np.concatenate([d.reshape(-1, I_HYPER) for d in listPostBurn])
    return {"daSplitRhat": fdaSplitRhat(listPostBurn).tolist(),
            "listPerRun": [dict(marginal.fdictSummarizeScatter(
                d["daPostBurn"].reshape(-1, I_HYPER), dictSummary),
                daTauSteps=d["daTau"].tolist(), dAcceptance=d["dAcceptance"])
                for d in listResults],
            "dictPooled": marginal.fdictSummarizeScatter(daPooled, dictSummary)}, daPooled


def main():
    """Run every independent run in parallel and save the comparison."""
    args = ftParseArguments()
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    listTasks = [(iRun, args.model, args.steps, args.walkers, args.thin, args.joint_sample)
                 for iRun in range(args.runs)]
    iProcesses = min(args.runs, max(1, multiprocessing.cpu_count() - 1))
    with multiprocessing.Pool(iProcesses) as pool:
        listResults = pool.map(fdictRunOne, listTasks)
    dictReport, daPooled = fdictSummarizeRuns(listResults, dictSummary)
    dictReport.update({"sModel": args.model, "iRuns": args.runs, "iSteps": args.steps,
                       "iWalkers": args.walkers})
    sStem = os.path.splitext(args.output)[0]
    np.save(sStem + "Chain.npy", daPooled)
    np.save(sStem + "ScatterTraces.npy", np.array([d["daScatterTrace"] for d in listResults]))
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
