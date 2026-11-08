#!/usr/bin/env python3
"""
Sample the marginalized joint-refit posterior with two independent algorithms.

dynesty nested sampling (Skilling 2006, Bayesian Analysis 1, 833; Speagle
2020, MNRAS 493, 3132) draws from the prior and contracts by volume, so its
result does not depend on where a sampler starts; its equally weighted draws
are the refit's primary product. Four independent emcee runs (Foreman-Mackey
et al. 2013) start from walkers dispersed over a wide box and run until every
run reaches 50 integrated autocorrelation times; agreement across runs is
measured with split-R-hat (Gelman et al. 2013, Bayesian Data Analysis, 3rd
ed., Section 11.4). Both samplers read the model from
dataRefitJointRelations.py, including its single prior specification.
"""

import multiprocessing

import numpy as np
from scipy import stats
import emcee
import dynesty

import dataRefitJointRelations as refit

I_LIVE_POINTS = 1000
D_STOP_DELTA_LOG_EVIDENCE = 0.1
D_IMPOSSIBLE = -1e300
I_EMCEE_RUNS = 4
I_EMCEE_WALKERS = 64
I_EMCEE_THIN = 10
I_EMCEE_CHUNK = 2000
I_EMCEE_MAX_STEPS = 80000
D_CONVERGENCE_FACTOR = 50.0
D_MAX_SPLIT_RHAT = 1.01
T_SCATTER_START_DEX = (0.003, 0.3)
LIST_MOVES = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
DICT_WORKER = {}


def fiCountProcesses():
    """Default worker count: every core but one."""
    return max(1, multiprocessing.cpu_count() - 1)


def fdaPriorTransform(daUnit, listPrior):
    """Map the unit cube to the prior described by refit.flistPriorSpecification."""
    return np.array([dFirst + (dSecond - dFirst) * dUnit if sKind == "uniform"
                     else stats.norm.ppf(dUnit, dFirst, dSecond)
                     for dUnit, (sKind, dFirst, dSecond) in zip(daUnit, listPrior)])


def fnInitializeNestedWorker(dictData):
    """Hold the model data in each nested-sampling worker process."""
    DICT_WORKER["dictData"] = dictData


def fdNestedLogLikelihood(daTheta):
    """Marginalized log-likelihood, with impossible points mapped to a finite floor."""
    dValue = refit.fdLogLikelihood(daTheta, DICT_WORKER["dictData"])
    return dValue if np.isfinite(dValue) else D_IMPOSSIBLE


def fdictRunNestedSampler(dictData, sSample, iSeed):
    """Run dynesty and return equally weighted draws plus diagnostics."""
    listPrior = refit.flistPriorSpecification(refit.DICT_PUBLISHED[sSample])
    iProcesses = fiCountProcesses()
    with multiprocessing.Pool(iProcesses, fnInitializeNestedWorker, (dictData,)) as pool:
        sampler = dynesty.NestedSampler(
            fdNestedLogLikelihood, fdaPriorTransform, len(listPrior),
            ptform_args=(listPrior,), nlive=I_LIVE_POINTS, sample="rslice",
            pool=pool, queue_size=iProcesses, rstate=np.random.default_rng(iSeed))
        sampler.run_nested(dlogz=D_STOP_DELTA_LOG_EVIDENCE, print_progress=False)
    results = sampler.results
    daWeights = np.exp(results.logwt - results.logz[-1])
    daChain = dynesty.utils.resample_equal(results.samples, daWeights / np.sum(daWeights),
                                           rstate=np.random.default_rng(iSeed))
    return {"daChain": daChain, "sSampler": "dynesty static, rslice",
            "iLivePoints": I_LIVE_POINTS, "iLikelihoodCalls": int(np.sum(results.ncall)),
            "dLogEvidence": float(results.logz[-1]),
            "dLogEvidenceError": float(results.logzerr[-1])}


def fdaDrawHyperStart(generator, sSample):
    """Draw one emcee start from a wide box around the published coefficients."""
    dictPub = refit.DICT_PUBLISHED[sSample]
    daRotation, daActivity = np.array(dictPub["daRotation"]), np.array(dictPub["daActivityRatio"])
    return np.concatenate([
        daRotation[:3] * generator.uniform(0.6, 1.4, 3),
        [generator.uniform(15.0, 40.0), generator.uniform(*np.log(T_SCATTER_START_DEX)),
         generator.uniform(-1.0, 1.0)],
        daActivity[:3] + generator.normal(0.0, [0.1, 0.2, 0.5]),
        [generator.uniform(-0.5, 0.8), np.log(generator.uniform(0.1, 1.0)),
         generator.uniform(-1.0, 1.0)]])


def fdaDrawDispersedStarts(iSeed, dictData, sSample):
    """Draw walker starts, redrawing any whose posterior is not finite."""
    generator = np.random.default_rng(iSeed)
    dictBounds = refit.DICT_PUBLISHED[sSample]
    listStarts = []
    while len(listStarts) < I_EMCEE_WALKERS:
        daCandidate = fdaDrawHyperStart(generator, sSample)
        if np.isfinite(refit.fdJointLogPosterior(daCandidate, dictData, dictBounds)):
            listStarts.append(daCandidate)
    return np.array(listStarts)


def fdictRunOneEmcee(tTask):
    """Run one emcee run until 50 autocorrelation times (or the step cap)."""
    dictData, sSample, iSeed = tTask
    np.random.seed(iSeed)
    sampler = emcee.EnsembleSampler(
        I_EMCEE_WALKERS, refit.I_HYPER_DIMENSIONS, refit.fdJointLogPosterior,
        args=(dictData, refit.DICT_PUBLISHED[sSample]), moves=LIST_MOVES)
    state, dTau = fdaDrawDispersedStarts(iSeed, dictData, sSample), np.inf
    while sampler.iteration * I_EMCEE_THIN < I_EMCEE_MAX_STEPS and \
            sampler.iteration * I_EMCEE_THIN < D_CONVERGENCE_FACTOR * dTau:
        state = sampler.run_mcmc(state, I_EMCEE_CHUNK // I_EMCEE_THIN,
                                 thin_by=I_EMCEE_THIN, progress=False)
        dTau = float(np.nan_to_num(np.max(sampler.get_autocorr_time(tol=0)),
                                   nan=np.inf)) * I_EMCEE_THIN
    daChain = sampler.get_chain()
    return {"daPostBurn": daChain[daChain.shape[0] // 2:], "iSteps": sampler.iteration * I_EMCEE_THIN,
            "dTauSteps": dTau, "dAcceptance": float(np.mean(sampler.acceptance_fraction))}


def fdaSplitRhat(listPostBurn):
    """Split-R-hat per parameter, treating each run's draws as one chain.

    Runs stop independently, so each contributes its final N draws, with N
    set by the shortest run, giving the equal-length chains R-hat requires.
    """
    listFlat = [daRun.reshape(-1, daRun.shape[-1], order="F") for daRun in listPostBurn]
    iDraws = min(len(daFlat) for daFlat in listFlat)
    listChains = []
    for daFlat in listFlat:
        daFlat = daFlat[-iDraws:]
        iHalf = len(daFlat) // 2
        listChains.extend([daFlat[:iHalf], daFlat[iHalf:2 * iHalf]])
    daChains = np.array(listChains)
    iLength = daChains.shape[1]
    daWithin = np.mean(np.var(daChains, axis=1, ddof=1), axis=0)
    daBetween = iLength * np.var(np.mean(daChains, axis=1), axis=0, ddof=1)
    return np.sqrt(((iLength - 1) / iLength * daWithin + daBetween / iLength) / daWithin)


def fdictRunEmceeRuns(dictData, sSample, iSeed):
    """Run the independent emcee runs in parallel; return pooled draws and gates."""
    listTasks = [(dictData, sSample, iSeed * 100 + iRun) for iRun in range(I_EMCEE_RUNS)]
    with multiprocessing.Pool(min(I_EMCEE_RUNS, fiCountProcesses())) as pool:
        listRuns = pool.map(fdictRunOneEmcee, listTasks)
    daRhat = fdaSplitRhat([d["daPostBurn"] for d in listRuns])
    bEveryRunLong = all(d["iSteps"] >= D_CONVERGENCE_FACTOR * d["dTauSteps"] for d in listRuns)
    dictDiagnostics = {
        "iRuns": I_EMCEE_RUNS, "iWalkers": I_EMCEE_WALKERS,
        "iaSteps": [d["iSteps"] for d in listRuns],
        "daTauSteps": [d["dTauSteps"] for d in listRuns],
        "daAcceptance": [d["dAcceptance"] for d in listRuns],
        "daSplitRhat": daRhat.tolist(), "dMaxSplitRhat": float(np.max(daRhat)),
        "bConverged": bool(bEveryRunLong and np.max(daRhat) < D_MAX_SPLIT_RHAT)}
    daPooled = np.concatenate([d["daPostBurn"].reshape(-1, refit.I_HYPER_DIMENSIONS)
                               for d in listRuns])
    return {"daChain": daPooled, "dictDiagnostics": dictDiagnostics}
