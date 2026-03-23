"""
Posterior inference for GJ 1132 XUV luminosity evolution (Ribas model).

Trains a GP surrogate of the VPLanet likelihood via alabi, then samples
posteriors with emcee, dynesty, PyMultiNest, and UltraNest. Reads MAP
parameters from MaxLEV output.

Usage:
    python dataBayesianPosteriors.py
    python dataBayesianPosteriors.py --sampler=emcee
    python dataBayesianPosteriors.py --lxuv-constraints path/to/lxuv_constraints.json

Prerequisites:
    Run MaxLEV first to generate MAP results:
        cd ../../EvolutionPlots/MaximumLikelihood
        maxlev gj1132_ribas.json --workers -1

Output (all in sSaveDir):
    surrogate_model.pkl       Trained GP surrogate
    emcee_samples.npz         Emcee posterior samples
    dynesty_samples.npz       Dynesty posterior samples
    multinest_samples.npz     PyMultiNest posterior samples
    ultranest_samples.npz     UltraNest posterior samples
"""

import argparse
import json
import multiprocessing
import os
import re
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from itertools import product
from pathlib import Path
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from sklearn import preprocessing

import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import astropy.units as u
import vplanet_inference as vpi

# =========================================================================
# Configuration
# =========================================================================

listParamLabels = [
    r"$m_{\star}$ [M$_{\odot}$]",
    r"$f_{sat}$",
    r"$t_{sat}$ [Gyr]",
    r"Age [Gyr]",
    r"$\beta_{XUV}$",
]

listBounds = [
    (0.17, 0.22),
    (-4.0, -2.15),
    (0.1, 5.0),
    (1.0, 13.0),
    (0.4, 2.1),
]

listPriorData = [
    (0.1945, 0.0048, 0.0046),
    (-2.92, 0.26),
    (None, None),
    "empirical",
    (1.18, 0.31),
]

I_NUM_DIMENSIONS = len(listBounds)

dictBaseGPConfig = {
    "fit_amp": True,
    "fit_mean": True,
    "fit_white_noise": False,
    "white_noise": -12,
    "uniform_scales": False,
    "hyperopt_method": "ml",
    "gp_opt_method": "l-bfgs-b",
    "gp_scale_rng": [-2, 6],
    "gp_amp_rng": [-1, 1],
    "multi_proc": True,
}

dictGPSearchGrid = {
    "kernel": ["ExpSquaredKernel", "Matern32Kernel", "Matern52Kernel"],
    "theta_scaler": [ut.no_scaler, preprocessing.MinMaxScaler(),
                     preprocessing.StandardScaler()],
    "y_scaler": [ut.no_scaler, ut.nlog_scaler, preprocessing.MinMaxScaler(),
                 preprocessing.StandardScaler()],
}

dictMLOptConfig = {
    "hyperopt_method": "ml",
    "optimizer_kwargs": {
        "maxiter": 100,
        "xatol": 1e-4,
        "fatol": 1e-3,
        "adaptive": True,
    },
}

dictActiveLearningConfig = {
    "algorithm": "bape",
    "gp_opt_freq": 50,
    "obj_opt_method": "nelder-mead",
    "use_grad_opt": False,
    "nopt": 1,
    "optimizer_kwargs": {
        "max_iter": 200,
        "xatol": 1e-3,
        "fatol": 1e-2,
        "adaptive": True,
    },
}

dictEmceeConfig = {
    "nwalkers": 20 * I_NUM_DIMENSIONS,
    "nsteps": int(2e4),
    "min_ess": int(1e4),
}

dictDynestyConfig = {
    "sampler_kwargs": {
        "bound": "multi",
        "nlive": 10 * I_NUM_DIMENSIONS,
        "sample": "rslice",
        "bootstrap": 0,
    },
    "run_kwargs": {
        "dlogz_init": 1.0,
        "print_progress": True,
    },
    "min_ess": int(1e4),
}

dictMultinestConfig = {
    "sampler_kwargs": {
        "n_live_points": 100 * I_NUM_DIMENSIONS,
        "sampling_efficiency": 0.8,
        "evidence_tolerance": 0.5,
    },
    "min_ess": int(1e4),
}

dictUltranestConfig = {
    "run_kwargs": {
        "min_num_live_points": 100 * I_NUM_DIMENSIONS,
        "dlogz": 0.5,
        "dKL": 0.5,
        "frac_remain": 0.01,
        "show_status": True,
    },
    "min_ess": int(1e4),
}

# Module-level state initialized by fnInitialize()
_daLikeData = None
_kdeAgePrior = None
_fnAgePriorInverseCDF = None
_vpm = None
_sInpath = None
_sSaveDir = None
_sMaxLevResultsPath = None
_iActiveIterations = None
_iNumCores = None


# =========================================================================
# Initialization
# =========================================================================


def fdaLoadLikeData(sConstraintsPath):
    """Load observational constraints from lxuv_constraints.json."""
    with open(sConstraintsPath, "r") as fileHandle:
        dictConstraint = json.load(fileHandle)
    return np.array([
        [4.38e-3, 3.4e-4],
        [dictConstraint["dMean"], dictConstraint["dStd"]],
    ])


def ftBuildEmpiricalPrior(sSamplesPath, tBounds):
    """Load age samples and return (KDE, inverse CDF interpolator).

    Samples are filtered to tBounds before building the KDE and
    inverse CDF.
    """
    daAgeGyr = np.loadtxt(sSamplesPath) / 1e9
    daAgeGyr = daAgeGyr[(daAgeGyr >= tBounds[0]) & (daAgeGyr <= tBounds[1])]
    kdeAge = gaussian_kde(daAgeGyr)
    daSorted = np.sort(daAgeGyr)
    daCDF = (np.arange(len(daSorted)) + 0.5) / len(daSorted)
    fnInverseCDF = interp1d(
        daCDF, daSorted, bounds_error=False,
        fill_value=(daSorted[0], daSorted[-1]),
    )
    return kdeAge, fnInverseCDF


def fnInitialize(sLxuvConstraintsPath, sAgeSamplesPath, sInpath,
                 sSaveDir, sMaxLevResultsPath):
    """Initialize all module-level state required for inference."""
    global _daLikeData, _kdeAgePrior, _fnAgePriorInverseCDF
    global _vpm, _sInpath, _sSaveDir, _sMaxLevResultsPath
    global _iActiveIterations, _iNumCores

    _sInpath = sInpath
    _sSaveDir = sSaveDir
    _sMaxLevResultsPath = sMaxLevResultsPath
    _iActiveIterations = 500
    _iNumCores = max(1, multiprocessing.cpu_count() - 1)

    _daLikeData = fdaLoadLikeData(sLxuvConstraintsPath)
    _kdeAgePrior, _fnAgePriorInverseCDF = ftBuildEmpiricalPrior(
        sAgeSamplesPath, listBounds[3])

    fnInitVplanetModel()


def fnInitVplanetModel():
    """Initialize the VPLanet forward model for GJ 1132."""
    global _vpm
    dictInparams = {
        "star.dMass": u.Msun,
        "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),
        "star.dSatXUVTime": u.Gyr,
        "vpl.dStopTime": u.Gyr,
        "star.dXUVBeta": u.dimensionless_unscaled,
    }
    dictOutparams = {
        "final.star.Luminosity": u.Lsun,
        "final.star.LXUVStellar": u.Lsun,
    }
    _vpm = vpi.VplanetModel(
        dictInparams, inpath=_sInpath, outparams=dictOutparams, verbose=False
    )


# =========================================================================
# Likelihood, Prior, and Posterior
# =========================================================================


def lnlike(daTheta):
    """Alias for fdLogLikelihood so pickle can resolve the old surrogate."""
    return fdLogLikelihood(daTheta)


def fdLogLikelihood(daTheta):
    """Compute log-likelihood by running VPLanet and comparing to data."""
    try:
        daOutput = _vpm.run_model(daTheta, remove=True)
    except Exception:
        return -1e2
    dLbol = daOutput[1]
    dLxuv = daOutput[0]
    if not (np.isfinite(dLbol) and np.isfinite(dLxuv)
            and dLbol > 0 and dLxuv > 0):
        return -1e2
    daModel = np.array([dLbol, np.log10(dLxuv / dLbol)])
    return -0.5 * np.sum(
        ((daModel - _daLikeData[:, 0]) / _daLikeData[:, 1]) ** 2)


def fdLogDensityEmpirical(dValue):
    """Evaluate log-density of the empirical age prior via KDE."""
    dDensity = _kdeAgePrior(dValue)[0]
    if dDensity <= 0:
        return -np.inf
    return float(np.log(dDensity))


def fdLogPrior(daTheta):
    """Evaluate log-prior with Gaussian, empirical, and uniform priors."""
    daTheta = np.atleast_1d(daTheta).flatten()
    for i, (dLower, dUpper) in enumerate(listBounds):
        if not (dLower <= float(daTheta[i]) <= dUpper):
            return -np.inf
    dLogPrior = 0.0
    for i, tPrior in enumerate(listPriorData):
        dLogPrior += fdLogPriorSingleDimension(
            float(daTheta[i]), tPrior)
    return dLogPrior


def fdLogPriorSingleDimension(dValue, tPrior):
    """Evaluate the log-prior contribution for one parameter dimension."""
    if tPrior == "empirical":
        return fdLogDensityEmpirical(dValue)
    if tPrior[0] is None:
        return 0.0
    if len(tPrior) == 2:
        return -0.5 * ((dValue - tPrior[0]) / tPrior[1]) ** 2
    dStd = tPrior[1] if dValue >= tPrior[0] else tPrior[2]
    return -0.5 * ((dValue - tPrior[0]) / dStd) ** 2


def fdaPriorTransform(daX):
    """Transform unit hypercube to parameter space with mixed priors."""
    daX = np.asarray(daX, dtype=float)
    daResult = np.zeros(I_NUM_DIMENSIONS)
    for i in range(I_NUM_DIMENSIONS):
        daResult[i] = fdPriorTransformSingleDimension(
            daX[i], listPriorData[i], i)
    return daResult


def fdPriorTransformSingleDimension(dX, tPrior, iDimension):
    """Transform a single unit-cube coordinate to parameter space."""
    if tPrior == "empirical":
        return float(_fnAgePriorInverseCDF(dX))
    if tPrior[0] is None:
        dLower, dUpper = listBounds[iDimension]
        return dLower + (dUpper - dLower) * dX
    return norm.ppf(dX, tPrior[0], tPrior[1])


def ffnSafeSurrogate(fnSurrogateFunction, dFloorValue=-1e4):
    """Clamp GP surrogate output to prevent extrapolation crashes."""
    def fnWrapper(daTheta):
        dValue = fnSurrogateFunction(daTheta)
        if not np.isfinite(dValue) or dValue > 0:
            return dFloorValue
        return max(dValue, dFloorValue)
    return fnWrapper


def ffnVarianceAwareSurrogate(fnSurrogateWithVariance, dFloorValue=-1e4,
                               dMaxVariance=100.0):
    """Clamp GP surrogate and reject high-variance extrapolations."""
    def fnWrapper(daTheta):
        dMean, dVariance = fnSurrogateWithVariance(daTheta)
        if not np.isfinite(dMean) or dMean > 0 or dVariance > dMaxVariance:
            return dFloorValue
        return max(dMean, dFloorValue)
    return fnWrapper


def fdaGenerateWalkerPositions(daMapParams, iNumWalkers):
    """Generate emcee walker starting positions centred on MAP."""
    iNumDim = len(daMapParams)
    daBoundWidths = np.array([dUpper - dLower
                              for dLower, dUpper in listBounds])
    daScatter = 0.01 * daBoundWidths
    daPositions = (daMapParams
                   + daScatter * np.random.randn(iNumWalkers, iNumDim))
    for i, (dLower, dUpper) in enumerate(listBounds):
        daPositions[:, i] = np.clip(daPositions[:, i], dLower, dUpper)
    return daPositions


# =========================================================================
# MaxLEV Results Reader
# =========================================================================

dictParamIndex = {
    "star.dMass": 0, "dMass": 0,
    "star.dSatXUVFrac": 1, "dSatXUVFrac": 1,
    "star.dSatXUVTime": 2, "dSatXUVTime": 2,
    "vpl.dStopTime": 3, "dStopTime": 3,
    "star.dXUVBeta": 4, "dXUVBeta": 4,
}


def ftReadMaxLevResults(sFilePath):
    """Read MAP parameters and objective value from MaxLEV results file.

    Returns (daParams, dNegLogPosterior) in canonical order.
    """
    daParams = np.full(I_NUM_DIMENSIONS, np.nan)
    dObjective = np.nan
    with open(sFilePath, "r") as fileHandle:
        for sLine in fileHandle:
            daParams, dObjective = ftParseSingleLine(
                sLine.strip(), daParams, dObjective)
    bMissing = np.isnan(daParams)
    if np.any(bMissing):
        listMissing = [sKey for sKey, iVal in dictParamIndex.items()
                       if "." in sKey and bMissing[iVal]]
        raise ValueError(
            f"Missing parameters in {sFilePath}: {listMissing}")
    return daParams, dObjective


def ftParseSingleLine(sStripped, daParams, dObjective):
    """Parse one line of a MaxLEV results file, updating in place."""
    matchParam = re.match(r"^(\S+)\s+=\s+(\S+)", sStripped)
    if matchParam:
        sName = matchParam.group(1)
        if sName in dictParamIndex:
            daParams[dictParamIndex[sName]] = float(matchParam.group(2))
    matchObjective = re.match(r"^-ln\(Posterior\)\s+=\s+(\S+)", sStripped)
    if matchObjective:
        dObjective = float(matchObjective.group(1))
    matchLikelihood = re.match(
        r"^-ln\(Likelihood\)\s+=\s+(\S+)", sStripped)
    if matchLikelihood and np.isnan(dObjective):
        dObjective = float(matchLikelihood.group(1))
    return daParams, dObjective


# =========================================================================
# Surrogate Model Training
# =========================================================================


def fiReadActiveIterations(sSaveDir):
    """Read completed active learning iterations from summary file."""
    sPath = os.path.join(sSaveDir, "surrogate_model.txt")
    try:
        with open(sPath, "r") as fileHandle:
            sContent = fileHandle.read()
        matchIter = re.search(
            r"Number of active training samples: (\d+)", sContent)
        return int(matchIter.group(1)) if matchIter else 0
    except FileNotFoundError:
        return 0


def flistDictCartesianProduct(dictOptions):
    """Generate all combinations of dict values as a list of dicts."""
    listKeys = list(dictOptions.keys())
    listValues = list(dictOptions.values())
    return [dict(zip(listKeys, tCombo))
            for tCombo in product(*listValues)]


def fdTestGPConfig(sm, dictConfig):
    """Evaluate a single GP configuration, returning test MSE or NaN."""
    try:
        return sm.init_gp(**dictConfig, overwrite=True)
    except Exception:
        return np.nan


def fsScalerName(scaler):
    """Return a human-readable name for a scaler object."""
    if hasattr(scaler, "name"):
        return scaler.name
    return type(scaler).__name__


def fdictSelectBestGPConfig(sm, dictBaseConfig, dictSearchGrid):
    """Grid-search kernel and scaler combinations by test MSE."""
    listCombinations = flistDictCartesianProduct(dictSearchGrid)
    iTotal = len(listCombinations)
    print(f"\nTesting {iTotal} GP hyperparameter combinations...")
    dBestMSE = np.inf
    dictBestConfig = None
    listResults = []
    for iIdx, dictVariable in enumerate(listCombinations):
        dictConfig = {**dictBaseConfig, **dictVariable}
        dTestMSE = fdTestGPConfig(sm, dictConfig)
        bIsBest = np.isfinite(dTestMSE) and dTestMSE < dBestMSE
        sSuffix = "  <-- best" if bIsBest else ""
        print(f"  [{iIdx+1}/{iTotal}] MSE={dTestMSE:.4e}{sSuffix}")
        listResults.append((dictVariable, dTestMSE))
        if bIsBest:
            dBestMSE = dTestMSE
            dictBestConfig = dictConfig
    if dictBestConfig is None:
        raise RuntimeError(
            "All GP configurations failed during grid search")
    print(f"\n  Best test MSE: {dBestMSE:.4e}")
    fnWriteGridSearchResults(listResults, dBestMSE)
    return dictBestConfig


def fnWriteGridSearchResults(listResults, dBestMSE):
    """Write grid search results to a text file."""
    sPath = os.path.join(_sSaveDir, "gp_grid_search_results.txt")
    with open(sPath, "w") as fileHandle:
        fileHandle.write(
            f"{'Kernel':<25s} {'Theta Scaler':<18s} "
            f"{'Y Scaler':<18s} {'Test MSE':<14s} {'Best'}\n")
        fileHandle.write("-" * 80 + "\n")
        for dictVariable, dMSE in listResults:
            fnWriteGridSearchRow(fileHandle, dictVariable, dMSE, dBestMSE)
    print(f"  Grid search results saved to {sPath}")


def fnWriteGridSearchRow(fileHandle, dictVariable, dMSE, dBestMSE):
    """Write a single row of the grid search results table."""
    sKernel = dictVariable["kernel"]
    sThetaScaler = fsScalerName(dictVariable["theta_scaler"])
    sYScaler = fsScalerName(dictVariable["y_scaler"])
    sBest = " <--" if dMSE == dBestMSE else ""
    fileHandle.write(
        f"{sKernel:<25s} {sThetaScaler:<18s} "
        f"{sYScaler:<18s} {dMSE:<14.4e}{sBest}\n")


def fnTrainSurrogate():
    """Train a new GP surrogate model from scratch."""
    print("\nTraining new surrogate model...")
    sm = SurrogateModel(
        lnlike_fn=fdLogLikelihood,
        bounds=listBounds,
        savedir=_sSaveDir,
        cache=True,
        verbose=True,
        show_warnings=True,
        ncore=_iNumCores,
        pool_method="fork",
    )
    iNumTraining = 600 * I_NUM_DIMENSIONS
    iNumTest = 100 * I_NUM_DIMENSIONS
    sm.init_samples(ntrain=iNumTraining, ntest=iNumTest, sampler="lhs")
    dictBestConfig = fdictSelectBestGPConfig(
        sm, dictBaseGPConfig, dictGPSearchGrid)
    sm.init_gp(**dictBestConfig, overwrite=True)
    sm.active_train(niter=_iActiveIterations, **dictActiveLearningConfig)
    return sm


def fsmLoadOrResumeSurrogate():
    """Load cached surrogate, resume partial training, or train."""
    sPklPath = os.path.join(_sSaveDir, "surrogate_model.pkl")
    if not os.path.exists(sPklPath):
        return fnTrainSurrogate()

    print(f"\nLoading cached surrogate from {_sSaveDir}...")
    sm = alabi.load_model_cache(_sSaveDir)
    iCompleted = fiReadActiveIterations(_sSaveDir)
    print(f"  Completed {iCompleted}/{_iActiveIterations} "
          f"active learning iterations")

    iRemaining = max(0, _iActiveIterations - iCompleted)
    if iRemaining > 0:
        print(f"  Resuming: {iRemaining} iterations remaining")
        sm.opt_gp_kwargs.update(dictMLOptConfig)
        sm.active_train(niter=iRemaining, **dictActiveLearningConfig)
    else:
        print("  Training complete, skipping to sampling")
    return sm


# =========================================================================
# Posterior Sampling
# =========================================================================


def fnRunEmcee(sm, daMapParams):
    """Run emcee posterior sampling on the trained surrogate."""
    import emcee

    sEmceePath = os.path.join(_sSaveDir, "emcee_samples.npz")
    if os.path.exists(sEmceePath):
        print(f"\nLoading cached emcee samples from {sEmceePath}...")
        sm.emcee_samples = np.load(sEmceePath)["samples"]
        print(f"  Emcee samples: {sm.emcee_samples.shape}")
        return
    print("\nRunning emcee...")
    fnSurrogate = sm.create_cached_surrogate_likelihood(
        iter=_iActiveIterations, return_var=False)
    fnSafeLikelihood = ffnSafeSurrogate(fnSurrogate)
    daWalkerPositions = fdaGenerateWalkerPositions(
        daMapParams, dictEmceeConfig["nwalkers"])
    sm.run_emcee(
        like_fn=fnSafeLikelihood,
        prior_fn=fdLogPrior,
        p0=daWalkerPositions,
        nwalkers=dictEmceeConfig["nwalkers"],
        nsteps=dictEmceeConfig["nsteps"],
        min_ess=dictEmceeConfig["min_ess"],
        multi_proc=False,
        samples_file="emcee_samples.npz",
        sampler_kwargs={
            "moves": [(emcee.moves.DEMove(), 0.8),
                      (emcee.moves.DESnookerMove(), 0.2)],
        },
    )
    print(f"  Emcee samples: {sm.emcee_samples.shape}")


def fnRunDynesty(sm):
    """Run dynesty nested sampling on the trained surrogate."""
    sDynestyPath = os.path.join(_sSaveDir, "dynesty_samples.npz")
    if os.path.exists(sDynestyPath):
        print(f"\nLoading cached dynesty samples from {sDynestyPath}...")
        sm.dynesty_samples = np.load(sDynestyPath)["samples"]
        print(f"  Dynesty samples: {sm.dynesty_samples.shape}")
        return
    print("\nRunning dynesty...")
    fnSurrogate = sm.create_cached_surrogate_likelihood(
        iter=_iActiveIterations)
    fnSafeLikelihood = ffnSafeSurrogate(fnSurrogate)
    sm.run_dynesty(
        like_fn=fnSafeLikelihood,
        prior_transform=fdaPriorTransform,
        sampler_kwargs=dictDynestyConfig["sampler_kwargs"],
        run_kwargs=dictDynestyConfig["run_kwargs"],
        min_ess=dictDynestyConfig["min_ess"],
        multi_proc=False,
        samples_file="dynesty_samples.npz",
    )
    print(f"  Dynesty samples: {sm.dynesty_samples.shape}")


def fnRunMultinest(sm):
    """Run PyMultiNest nested sampling on the trained surrogate."""
    sMultinestPath = os.path.join(_sSaveDir, "multinest_samples.npz")
    if os.path.exists(sMultinestPath):
        print(f"\nLoading cached multinest samples...")
        sm.pymultinest_samples = np.load(sMultinestPath)["samples"]
        print(f"  MultiNest samples: {sm.pymultinest_samples.shape}")
        return
    print("\nRunning PyMultiNest...")
    fnSurrogate = sm.create_cached_surrogate_likelihood(
        iter=_iActiveIterations)
    fnSafeLikelihood = ffnSafeSurrogate(fnSurrogate)
    sm.run_pymultinest(
        like_fn=fnSafeLikelihood,
        prior_transform=fdaPriorTransform,
        sampler_kwargs=dictMultinestConfig["sampler_kwargs"],
        min_ess=dictMultinestConfig["min_ess"],
        multi_proc=False,
        samples_file="multinest_samples.npz",
    )
    print(f"  MultiNest samples: {sm.pymultinest_samples.shape}")


def fnRunUltranest(sm):
    """Run UltraNest nested sampling on the trained surrogate."""
    sUltranestPath = os.path.join(_sSaveDir, "ultranest_samples.npz")
    if os.path.exists(sUltranestPath):
        print(f"\nLoading cached ultranest samples...")
        sm.ultranest_samples = np.load(sUltranestPath)["samples"]
        print(f"  UltraNest samples: {sm.ultranest_samples.shape}")
        return
    print("\nRunning UltraNest...")
    fnSurrogate = sm.create_cached_surrogate_likelihood(
        iter=_iActiveIterations)
    fnSafeLikelihood = ffnSafeSurrogate(fnSurrogate)
    sm.run_ultranest(
        like_fn=fnSafeLikelihood,
        prior_transform=fdaPriorTransform,
        run_kwargs=dictUltranestConfig["run_kwargs"],
        min_ess=dictUltranestConfig["min_ess"],
        samples_file="ultranest_samples.npz",
        resume="overwrite",
    )
    print(f"  UltraNest samples: {sm.ultranest_samples.shape}")


# =========================================================================
# Summary Statistics
# =========================================================================


def fsQualifyAgreement(dDeltaSigma):
    """Return a qualitative label for a delta/sigma agreement."""
    if dDeltaSigma < 0.5:
        return "excellent"
    if dDeltaSigma < 1.0:
        return "good"
    if dDeltaSigma < 2.0:
        return "moderate"
    return "poor"


def fnPrintPosteriorComparison(listSamplerResults):
    """Print pairwise comparison statistics for all samplers."""
    print("\n" + "=" * 70)
    print("POSTERIOR COMPARISON")
    print("=" * 70)
    for i, sLabel in enumerate(listParamLabels):
        print(f"\n{sLabel}:")
        fnPrintSingleParamComparison(listSamplerResults, i)
    print("\n" + "=" * 70)


def fnPrintSingleParamComparison(listSamplerResults, iParamIndex):
    """Print per-sampler stats and pairwise agreement for one parameter."""
    for sName, daSamples in listSamplerResults:
        dMean = np.mean(daSamples[:, iParamIndex])
        dStd = np.std(daSamples[:, iParamIndex])
        print(f"  {sName:12s}: {dMean:.6f} +/- {dStd:.6f}")
    iNumSamplers = len(listSamplerResults)
    for iA in range(iNumSamplers):
        for iB in range(iA + 1, iNumSamplers):
            fnPrintPairwiseAgreement(
                listSamplerResults[iA], listSamplerResults[iB],
                iParamIndex)


def fnPrintPairwiseAgreement(tSamplerA, tSamplerB, iParamIndex):
    """Print delta/sigma agreement between two samplers."""
    sNameA, daSamplesA = tSamplerA
    sNameB, daSamplesB = tSamplerB
    dMeanA = np.mean(daSamplesA[:, iParamIndex])
    dStdA = np.std(daSamplesA[:, iParamIndex])
    dMeanB = np.mean(daSamplesB[:, iParamIndex])
    dStdB = np.std(daSamplesB[:, iParamIndex])
    dDeltaSigma = abs(dMeanA - dMeanB) / ((dStdA + dStdB) / 2)
    sQuality = fsQualifyAgreement(dDeltaSigma)
    print(f"  {sNameA} vs {sNameB}: "
          f"delta/sigma={dDeltaSigma:.2f} ({sQuality})")


def fnSaveVspaceTransform(daSamples, sOutputPath):
    """Convert dynesty posterior samples to vspace-compatible format."""
    daTransformed = daSamples.copy()
    daTransformed[:, 1] = 10.0 ** daSamples[:, 1]
    daTransformed[:, 2] = daSamples[:, 2] * 1e9
    daTransformed[:, 3] = daSamples[:, 3] * 1e9
    np.save(sOutputPath, daTransformed)
    print(f"  Saved vspace transform ({daTransformed.shape[0]} samples)"
          f" to {sOutputPath}")


# =========================================================================
# Main
# =========================================================================


def fdaReadMapParameters():
    """Read MAP parameters from MaxLEV results file."""
    if os.path.exists(_sMaxLevResultsPath):
        print(f"\nReading MAP results from {_sMaxLevResultsPath}...")
        daMaxLikeParams, dNegLogPost = ftReadMaxLevResults(
            _sMaxLevResultsPath)
        print(f"  MAP parameters: {daMaxLikeParams}")
        print(f"  -ln(Posterior): {dNegLogPost:.6e}")
        return daMaxLikeParams
    print(f"\nWARNING: MaxLEV results not found at {_sMaxLevResultsPath}")
    return None


def fnRunSelectedSamplers(sm, daMaxLikeParams, sSamplerOnly):
    """Run either all samplers or a single named sampler."""
    bRunAll = sSamplerOnly is None
    if bRunAll or sSamplerOnly == "emcee":
        if daMaxLikeParams is None:
            raise RuntimeError(
                "MAP parameters required for emcee walker "
                "initialization. Run MaxLEV first.")
        fnRunEmcee(sm, daMaxLikeParams)
    if bRunAll or sSamplerOnly == "dynesty":
        fnRunDynesty(sm)
    if bRunAll or sSamplerOnly == "multinest":
        fnRunMultinest(sm)
    if bRunAll or sSamplerOnly == "ultranest":
        fnRunUltranest(sm)


def ftParseArguments():
    """Parse CLI arguments for the data inference script."""
    parser = argparse.ArgumentParser(
        description="GJ 1132 Bayesian posterior inference.")
    parser.add_argument(
        '--sampler', default=None,
        choices=['emcee', 'dynesty', 'multinest', 'ultranest'],
        help="Run only the specified sampler.")
    parser.add_argument(
        '--lxuv-constraints', default=None, metavar='PATH',
        help="Path to lxuv_constraints.json.")
    return parser.parse_args()


def fnMain():
    """Run posterior inference pipeline."""
    args = ftParseArguments()

    sInpath = os.path.dirname(os.path.abspath(__file__))
    sSaveDir = "output/"
    sMaxLevResultsPath = str(
        Path(sInpath).parent / "MaximumLikelihood" / "maxlike_results.txt")
    sAgeSamplesPath = str(
        Path(sInpath).parent / "EngleAgeDistribution" / "age_samples.txt")

    if args.lxuv_constraints:
        sLxuvConstraintsPath = args.lxuv_constraints
    else:
        sLxuvConstraintsPath = os.path.join(sInpath, "lxuv_constraints.json")

    print("=" * 70)
    print("GJ 1132 Ribas XUV Model - Posterior Inference")
    print("=" * 70)

    os.makedirs(sSaveDir, exist_ok=True)
    fnInitialize(sLxuvConstraintsPath, sAgeSamplesPath, sInpath,
                 sSaveDir, sMaxLevResultsPath)

    daMaxLikeParams = fdaReadMapParameters()
    sm = fsmLoadOrResumeSurrogate()

    fnRunSelectedSamplers(sm, daMaxLikeParams, args.sampler)

    if args.sampler is None:
        listSamplerResults = [
            ("Emcee", sm.emcee_samples),
            ("Dynesty", sm.dynesty_samples),
            ("MultiNest", sm.pymultinest_samples),
            ("UltraNest", sm.ultranest_samples),
        ]
        fnPrintPosteriorComparison(listSamplerResults)

    sTransformPath = os.path.join(
        sSaveDir, "dynesty_transform_final.npy")
    fnSaveVspaceTransform(sm.dynesty_samples, sTransformPath)

    print("\nDone.")


if __name__ == "__main__":
    fnMain()
