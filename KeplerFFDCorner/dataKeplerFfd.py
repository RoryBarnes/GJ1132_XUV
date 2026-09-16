#!/usr/bin/env python3
"""
Bayesian inference of stellar flare frequency distribution parameters for ensemble data.
Based on Davenport et al. 2019 (ApJ 871, 241), Equation 3.

This script uses emcee to infer posterior distributions for parameters
a1, a2, a3, b1, b2, b3 in the flare frequency distribution model using
data from multiple stars in an ensemble.

CORRECTED PARAMETER DEFINITIONS:
From Davenport et al. 2019 Eq. (3):
log10(rate) = (a1*log_age + a2*mass + a3) * log_energy + (b1*log_age + b2*mass + b3)

Where:
- a1, a2, a3: Control the energy slope (power-law index dependence)  
- b1, b2, b3: Control the normalization (intercept dependence)
"""

import json
import os

import numpy as np
import pandas as pd
import emcee
from scipy.optimize import differential_evolution

def fdaLogFlareRateModel(daLogEnergy, daLogAge, daMass, daParams):
    """
    Model for log10(flare rate) based on Davenport et al. 2019 Equation (3).

    From Eq. (3): log10(rate) = (a1*log_age + a2*mass + a3) * log_energy + (b1*log_age + b2*mass + b3)

    Where the "a" terms control the slope (energy dependence) and "b" terms control the intercept.
    """
    dA1, dA2, dA3, dB1, dB2, dB3 = daParams
    dSlope = dA1 * daLogAge + dA2 * daMass + dA3
    dIntercept = dB1 * daLogAge + dB2 * daMass + dB3
    daLogRate = dSlope * daLogEnergy + dIntercept
    return daLogRate

def fdLogLikelihoodEnsemble(daParams, daLogEnergy, daLogAge, daMass,
                            daObservedLogFf, daLogFfErrors):
    """Log-likelihood for the ensemble flare frequency model."""
    daPredictedLogFf = fdaLogFlareRateModel(
        daLogEnergy, daLogAge, daMass, daParams
    )
    daResiduals = daObservedLogFf - daPredictedLogFf
    dChiSquared = np.sum((daResiduals / daLogFfErrors) ** 2)
    dLogLike = -0.5 * (
        dChiSquared + np.sum(np.log(2 * np.pi * daLogFfErrors ** 2))
    )
    if not np.isfinite(dLogLike):
        return -np.inf
    return dLogLike

def fdLogPriorEnsemble(daParams):
    """Log-prior with physically motivated bounds for ensemble analysis."""
    dA1, dA2, dA3, dB1, dB2, dB3 = daParams
    if (-5 < dA1 < 5 and
        -5 < dA2 < 5 and
        -3 < dA3 < 0 and
        -20 < dB1 < 20 and
        -20 < dB2 < 20 and
        -30 < dB3 < 30):
        return 0.0
    return -np.inf

def fdLogPosteriorEnsemble(daParams, daLogEnergy, daLogAge, daMass,
                           daObservedLogFf, daLogFfErrors):
    """Log-posterior probability for ensemble analysis."""
    dLogPrior = fdLogPriorEnsemble(daParams)
    if not np.isfinite(dLogPrior):
        return -np.inf
    dLogLike = fdLogLikelihoodEnsemble(
        daParams, daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors
    )
    return dLogPrior + dLogLike

def fdaFindInitialGuess(daLogEnergy, daLogAge, daMass,
                       daObservedLogFf, daLogFfErrors):
    """Find initial parameter guess via maximum likelihood estimation."""

    def fdNegLogLikelihood(daParams):
        return -fdLogLikelihoodEnsemble(
            daParams, daLogEnergy, daLogAge, daMass,
            daObservedLogFf, daLogFfErrors
        )

    daDefault = np.array([0.0, 0.0, -1.5, -2.0, 5.0, 15.0])
    listBounds = [(-5, 5), (-5, 5), (-3, 0),
                  (-20, 20), (-20, 20), (-30, 30)]
    return fdaOptimizeInitialParams(fdNegLogLikelihood, listBounds, daDefault)


def fdaOptimizeInitialParams(fdObjective, listBounds, daDefault):
    """Run differential evolution and return best-fit parameters."""
    print("Optimizing initial parameters...")
    try:
        result = differential_evolution(
            fdObjective, listBounds, seed=42, maxiter=500
        )
        if result.success:
            print(f"Optimization successful. Likelihood: {-result.fun:.2f}")
        else:
            print("Warning: Optimization did not fully converge.")
        return result.x
    except Exception as error:
        print(f"Optimization failed: {error}")
        print("Using default initial guess")
        return daDefault

def fdataLoadEnsembleData(sDataFile):
    """Load ensemble FFD CSV data. Raises FileNotFoundError if missing."""
    print("Loading ensemble data...")
    if not os.path.exists(sDataFile):
        raise FileNotFoundError(
            f"Required data file not found: {sDataFile}\n"
            f"Copy ensemble_FFD.csv into this directory before running."
        )
    dfData = pd.read_csv(sDataFile)
    print(f"Loaded {len(dfData)} rows from {sDataFile}")
    return dfData


def fdataFilterValidRows(dfData):
    """Filter to rows with finite, positive flare frequency values."""
    baMask = (
        (dfData['FF'] > 0) &
        np.isfinite(dfData['FF']) &
        np.isfinite(dfData['FFerr']) &
        (dfData['FFerr'] > 0) &
        np.isfinite(dfData['logAge']) &
        np.isfinite(dfData['mass']) &
        np.isfinite(dfData['logE'])
    )
    dfClean = dfData[baMask].copy()
    if len(dfClean) == 0:
        raise ValueError("No valid flare frequency data found!")
    return dfClean


def ftConvertToLogSpace(dfClean):
    """Extract arrays and convert flare frequencies to log space."""
    daLogEnergy = dfClean['logE'].values
    daLogAge = dfClean['logAge'].values
    daMass = dfClean['mass'].values
    daObservedFf = dfClean['FF'].values
    daFfErrors = dfClean['FFerr'].values
    daObservedLogFf = np.log10(daObservedFf)
    daLogFfErrors = daFfErrors / (daObservedFf * np.log(10))
    dMinLogError = 0.01
    daLogFfErrors = np.maximum(daLogFfErrors, dMinLogError)
    return daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors


def fnPrintDataRangeWarnings(daLogAge, daMass):
    """Print warnings if age or mass ranges are too narrow."""
    dAgeRange = daLogAge.max() - daLogAge.min()
    dMassRange = daMass.max() - daMass.min()
    if dAgeRange < 0.1:
        print(f"Warning: Limited age range ({dAgeRange:.3f} dex).")
    if dMassRange < 0.1:
        print(f"Warning: Limited mass range ({dMassRange:.2f} M_sun).")


def fnPrintEnsembleStatistics(dfClean, daLogEnergy, daLogAge, daMass,
                              daObservedLogFf, iaStarIds):
    """Print summary statistics for the loaded ensemble data."""
    iNumStars = len(np.unique(iaStarIds))
    iNumPoints = len(dfClean)
    print(f"Ensemble statistics:")
    print(f"  Total valid data points: {iNumPoints}")
    print(f"  Number of stars: {iNumStars}")
    print(f"  Average points per star: {iNumPoints/iNumStars:.1f}")
    print(f"Data ranges:")
    print(f"  Energy: {daLogEnergy.min():.2f} to {daLogEnergy.max():.2f}")
    print(f"  Age: {daLogAge.min():.3f} to {daLogAge.max():.3f}")
    print(f"  Mass: {daMass.min():.2f} to {daMass.max():.2f}")
    print(f"  Flare rate: {daObservedLogFf.min():.2f} to "
          f"{daObservedLogFf.max():.2f}")
    fnPrintDataRangeWarnings(daLogAge, daMass)


def ftLoadEnsembleData(sDataFile):
    """Load and process ensemble flare frequency data."""
    dfData = fdataLoadEnsembleData(sDataFile)
    print("Processing ensemble data...")
    dfClean = fdataFilterValidRows(dfData)
    listStellarProps = ['logAge', 'mass', 'Prot']
    dfClean['star_id'] = dfClean.groupby(listStellarProps).ngroup()
    iaStarIds = dfClean['star_id'].values
    daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors = (
        ftConvertToLogSpace(dfClean)
    )
    fnPrintEnsembleStatistics(
        dfClean, daLogEnergy, daLogAge, daMass, daObservedLogFf, iaStarIds
    )
    return daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors, iaStarIds

def fdaInitializeWalkerPositions(daInitialParams, iNumWalkers, iNumDim):
    """Initialize walker positions within prior bounds."""
    daPositions = daInitialParams + 0.1 * np.random.randn(iNumWalkers, iNumDim)
    for i in range(iNumWalkers):
        while fdLogPriorEnsemble(daPositions[i]) == -np.inf:
            daPositions[i] = daInitialParams + 0.1 * np.random.randn(iNumDim)
    return daPositions


def fnCheckConvergence(sampler, iNumSteps):
    """Check and report MCMC chain convergence."""
    try:
        daTau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: {daTau}")
        if np.any(daTau * 50 > iNumSteps):
            print("Warning: Chain may not be converged.")
        else:
            print("Chain appears well converged.")
    except Exception as error:
        print(f"Could not compute autocorrelation time: {error}")


def ftCreateAndRunSampler(daPositions, iNumWalkers, iNumDim,
                          tSamplerArgs, iBurnIn, iNumSteps):
    """Create emcee sampler, run burn-in and production chain."""
    sampler = emcee.EnsembleSampler(
        iNumWalkers, iNumDim, fdLogPosteriorEnsemble, args=tSamplerArgs
    )
    print(f"Running burn-in with {iBurnIn} steps...")
    daPositions, _, _ = sampler.run_mcmc(daPositions, iBurnIn, progress=True)
    sampler.reset()
    print(f"Running MCMC with {iNumSteps} steps...")
    sampler.run_mcmc(daPositions, iNumSteps, progress=True)
    return sampler


def fnRunMcmcEnsemble(sDataFile, iNumWalkers=32, iNumSteps=5000,
                      iBurnIn=1000, iThin=10):
    """Run MCMC sampling to infer flare frequency parameters."""
    daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors, _ = (
        ftLoadEnsembleData(sDataFile)
    )
    print("Finding initial parameter guess...")
    daInitialParams = fdaFindInitialGuess(
        daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors
    )
    iNumDim = 6
    daPositions = fdaInitializeWalkerPositions(
        daInitialParams, iNumWalkers, iNumDim
    )
    tArgs = (daLogEnergy, daLogAge, daMass, daObservedLogFf, daLogFfErrors)
    sampler = ftCreateAndRunSampler(
        daPositions, iNumWalkers, iNumDim, tArgs, iBurnIn, iNumSteps
    )
    fnCheckConvergence(sampler, iNumSteps)
    return sampler

def fsSaveSamples(daSamples, listParamNames,
                  sFilename='flare_mcmc_samples.txt'):
    """Save MCMC samples to a tab-delimited text file with header."""
    print(f"Saving {len(daSamples)} MCMC samples to {sFilename}...")
    sHeader = (
        "MCMC samples from stellar flare frequency analysis\n"
        "Based on Davenport et al. 2019 (ApJ 871, 241) Equation (3)\n"
        f"Columns: {' '.join(listParamNames)}\n"
        f"Number of samples: {len(daSamples)}\n"
        "  a1, a2, a3: Control energy slope (power-law index)\n"
        "  b1, b2, b3: Control normalization (intercept)\n"
    )
    np.savetxt(sFilename, daSamples, header=sHeader,
               fmt='%.6f', delimiter='\t')
    print(f"File: {daSamples.shape[1]} params, {daSamples.shape[0]} samples")
    return sFilename

def fnSaveParamNames(listParamNames, sFilename):
    """Write parameter name index file."""
    with open(sFilename, 'w') as fileHandle:
        fileHandle.write("# Parameter names for MCMC samples\n")
        fileHandle.write("# From Davenport et al. 2019 Eq. (3)\n")
        for i, sName in enumerate(listParamNames):
            fileHandle.write(f"{i}\t{sName}\n")


def fnSaveSamplesCsv(daSamples, listParamNames, sFilename):
    """Save MCMC samples as CSV."""
    print(f"Saving samples to {sFilename}...")
    dfSamples = pd.DataFrame(daSamples, columns=listParamNames)
    dfSamples.to_csv(sFilename, index=False, float_format='%.6f')


def fdictSaveMultipleFormats(daSamples, listParamNames,
                             sBaseFilename='flare_mcmc_samples'):
    """Save MCMC samples in text, CSV, NumPy, and index formats."""
    dictFilenames = {}
    dictFilenames['txt'] = fsSaveSamples(
        daSamples, listParamNames, f"{sBaseFilename}.txt"
    )
    fnSaveSamplesCsv(daSamples, listParamNames, f"{sBaseFilename}.csv")
    dictFilenames['csv'] = f"{sBaseFilename}.csv"
    sNpyFile = f"{sBaseFilename}.npy"
    print(f"Saving samples to {sNpyFile}...")
    np.save(sNpyFile, daSamples)
    dictFilenames['npy'] = sNpyFile
    sParamsFile = f"{sBaseFilename}_param_names.txt"
    fnSaveParamNames(listParamNames, sParamsFile)
    dictFilenames['params'] = sParamsFile
    return dictFilenames
def fdictComputeStatistics(daSamples, listParamNames):
    """Compute median and 1-sigma uncertainties for each parameter."""
    dictStats = {}
    for iParam, sName in enumerate(listParamNames):
        daPercentiles = np.percentile(
            daSamples[:, iParam], [16, 50, 84]
        )
        daUncertainty = np.diff(daPercentiles)
        dictStats[sName] = {
            "fMedian": float(daPercentiles[1]),
            "fUpperSigma": float(daUncertainty[1]),
            "fLowerSigma": float(daUncertainty[0]),
        }
    return dictStats


def fnWritePosteriorStatistics(daSamples, sOutputPath):
    """Write posterior medians and full covariance matrix to JSON."""
    daMedians = np.median(daSamples, axis=0).tolist()
    daCovarianceMatrix = np.cov(daSamples, rowvar=False).tolist()
    listParamNames = ["a1", "a2", "a3", "b1", "b2", "b3"]

    dictStatistics = {
        "daMedians": daMedians,
        "daCovarianceMatrix": daCovarianceMatrix,
        "listParamNames": listParamNames,
        "iNumSamples": daSamples.shape[0],
    }
    with open(sOutputPath, "w") as fileHandle:
        json.dump(dictStatistics, fileHandle, indent=2)
    print(f"Posterior statistics saved to: {sOutputPath}")


def fdaAnalyzeResults(sampler, iThin=10):
    """Extract samples, save data files, and print statistics."""
    daSamples = sampler.get_chain(discard=0, thin=iThin, flat=True)
    listParamNames = ["a1", "a2", "a3", "b1", "b2", "b3"]
    fdictSaveMultipleFormats(daSamples, listParamNames)
    dictStats = fdictComputeStatistics(daSamples, listParamNames)
    for sName, dictValues in dictStats.items():
        print(
            f"{sName}: {dictValues['fMedian']:.4f} "
            f"+{dictValues['fUpperSigma']:.4f} "
            f"-{dictValues['fLowerSigma']:.4f}"
        )
    sScriptDir = os.path.dirname(os.path.abspath(__file__))
    fnWritePosteriorStatistics(
        daSamples,
        os.path.join(sScriptDir, "kepler_ffd_posterior_stats.json"),
    )
    return daSamples


if __name__ == "__main__":
    sScriptDirectory = os.path.dirname(os.path.abspath(__file__))
    sCachedSamples = os.path.join(
        sScriptDirectory, "flare_mcmc_samples.npy"
    )

    if os.path.exists(sCachedSamples):
        print("Loading cached MCMC samples...")
        daSamples = np.load(sCachedSamples)
        print(f"Loaded {daSamples.shape[0]} samples")
        fnWritePosteriorStatistics(
            daSamples,
            os.path.join(sScriptDirectory, "kepler_ffd_posterior_stats.json"),
        )
    else:
        np.random.seed(42)
        print("Stellar Flare Frequency Ensemble MCMC Analysis")
        print("=" * 50)
        sampler = fnRunMcmcEnsemble(
            "ensemble_FFD.csv", iNumWalkers=32, iNumSteps=8000, iBurnIn=1000
        )
        daSamples = fdaAnalyzeResults(sampler)

    print("\nData analysis complete!")
    print("Run plotKeplerFfd.py to generate figures.")