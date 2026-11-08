#!/usr/bin/env python3
"""
Sample the marginalized A08 mid-late model with dynesty nested sampling.

The likelihood is the A08 model with benchmark latent ages marginalized
(explorations/refitMarginalizedLatentAges.py). Nested sampling (Skilling 2006,
Bayesian Analysis 1, 833; dynesty: Speagle 2020, MNRAS 493, 3132) draws from
the prior and contracts by volume, so it does not depend on where walkers
start. The prior transform reproduces utils/dataRefitJointRelations
.fdHyperLogPrior term by term, and that equivalence is checked numerically
before sampling.

Usage: python runNestedSamplingMarginalized.py --joint-sample <json>
       --fit-summary <json> --output <json>
"""

import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
from scipy import stats
import dynesty

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S_DIRECTORY)
import refitMarginalizedLatentAges as marginal  # noqa: E402
from utils import dataRefitJointRelations as refit  # noqa: E402

I_SEED = 42
I_LIVE_POINTS = 1000
D_STOP_DELTA_LOG_EVIDENCE = 0.1
D_IMPOSSIBLE = -1e300
DICT_DATA = {}
LIST_PRIOR = [("normal", 0.0, 0.1), ("normal", 0.0, 5.0), ("normal", 0.0, 0.1),
              ("uniform", 5.0, 80.0), ("normal", np.log(0.15), 1.5), ("normal", 0.0, 2.0),
              ("normal", 0.0, 1.0), ("normal", -3.5, 2.0), ("normal", 0.0, 3.0),
              ("uniform", -1.2, 1.2), ("normal", np.log(0.30), 1.5), ("normal", 0.0, 2.0)]


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for sName in ("joint-sample", "fit-summary", "output"):
        parser.add_argument(f"--{sName}", required=True)
    return parser.parse_args()


def fdaPriorTransform(daUnit):
    """Map the unit cube to the A08 hyperparameter prior."""
    return np.array([dLow + (dHigh - dLow) * dUnit if sKind == "uniform"
                     else stats.norm.ppf(dUnit, dLow, dHigh)
                     for dUnit, (sKind, dLow, dHigh) in zip(daUnit, LIST_PRIOR)])


def fdLogLikelihood(daTheta):
    """Marginalized A08 log-likelihood (no prior term)."""
    dBenchmark = marginal.fdBenchmarkMarginalLogLike(daTheta, DICT_DATA["dictData"])
    if not np.isfinite(dBenchmark):
        return D_IMPOSSIBLE
    dTotal = dBenchmark + marginal.fdNonBenchmarkLogLike(daTheta, DICT_DATA["dictData"])
    return dTotal if np.isfinite(dTotal) else D_IMPOSSIBLE


def fnInitializeWorker(sJointSample):
    """Build the model data once in each worker process."""
    DICT_DATA["dictData"] = refit.fdictBuildData(sJointSample, "midlate", True)


def fdCheckPriorEquivalence():
    """Return the spread of (transform log-density - A08 log-prior) over draws."""
    generator = np.random.default_rng(I_SEED)
    listDifferences = []
    for _ in range(200):
        daTheta = fdaPriorTransform(generator.uniform(0, 1, len(LIST_PRIOR)))
        dTransform = sum(stats.norm.logpdf(dValue, dLow, dHigh) for dValue, (sKind, dLow, dHigh)
                         in zip(daTheta, LIST_PRIOR) if sKind == "normal")
        listDifferences.append(dTransform - refit.fdHyperLogPrior(
            daTheta, refit.DICT_PUBLISHED["midlate"]))
    return float(np.ptp(listDifferences))


def ftRunNestedSampler(sJointSample):
    """Run dynesty in parallel and return equally weighted posterior draws."""
    iProcesses = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(iProcesses, fnInitializeWorker, (sJointSample,)) as pool:
        sampler = dynesty.NestedSampler(
            fdLogLikelihood, fdaPriorTransform, len(LIST_PRIOR), nlive=I_LIVE_POINTS,
            sample="rslice", pool=pool, queue_size=iProcesses,
            rstate=np.random.default_rng(I_SEED))
        sampler.run_nested(dlogz=D_STOP_DELTA_LOG_EVIDENCE, print_progress=False)
    results = sampler.results
    daWeights = np.exp(results.logwt - results.logz[-1])
    return dynesty.utils.resample_equal(results.samples, daWeights / np.sum(daWeights),
                                        rstate=np.random.default_rng(I_SEED)), results


def main():
    """Check the prior, run dynesty, and summarize the posterior."""
    args = ftParseArguments()
    with open(args.fit_summary) as fileHandle:
        dictSummary = json.load(fileHandle)
    dPriorMismatch = fdCheckPriorEquivalence()
    print(f"prior transform vs A08 prior: spread of difference {dPriorMismatch:.2e}", flush=True)
    if dPriorMismatch > 1e-8:
        sys.exit("ERROR: prior transform does not reproduce the A08 prior; not sampling.")
    daPosterior, results = ftRunNestedSampler(args.joint_sample)
    dictReport = {"dPriorMismatch": dPriorMismatch, "iLikelihoodCalls": int(np.sum(results.ncall)),
                  "dLogEvidence": float(results.logz[-1]),
                  "dLogEvidenceError": float(results.logzerr[-1]),
                  "iPosteriorDraws": int(len(daPosterior)),
                  "dictPosterior": marginal.fdictSummarizeScatter(daPosterior, dictSummary)}
    np.save(os.path.splitext(args.output)[0] + "Chain.npy", daPosterior)
    with open(args.output, "w") as fileHandle:
        json.dump(dictReport, fileHandle, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
