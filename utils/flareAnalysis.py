"""Shared functions for TESS flare analysis of GJ 1132.

This module contains data loading, FFD fitting, literature data retrieval,
and statistical comparison functions used by multiple plotting scripts.
"""

import json
import os

import numpy as np
import astropy.constants as const
import lightkurve as lk
from scipy.optimize import curve_fit
import vplot

from utils.ffd import FFD


# ---------------------------------------------------------------------------
# FFD fitting
# ---------------------------------------------------------------------------

def fdFfdFit(daLogEnergy, dAlpha, dBeta):
    """Return log flare rate from a linear FFD model."""
    return dBeta + (daLogEnergy * dAlpha)


def fdComputeReducedChi2(daResiduals, daErrors, iNumParams):
    """Return reduced chi-squared from residuals and uncertainties."""
    dChi2 = np.sum((daResiduals / daErrors) ** 2)
    iDof = len(daResiduals) - iNumParams
    dReducedChi2 = dChi2 / iDof if iDof > 0 else np.nan
    return dChi2, iDof, dReducedChi2


def fdictComputeFfdBestfit(daFfdX, daFfdY, daFfdYerr):
    """Fit a power-law FFD and return parameter estimates.

    Returns a dict with keys alpha, alpha_err, beta, beta_err,
    popt, pcov, chi2, dof, and reduced_chi2.
    """
    daInitialGuess = [-1.0, 30.0]
    daPopt, daPcov = curve_fit(
        fdFfdFit, daFfdX, daFfdY, p0=daInitialGuess,
        sigma=daFfdYerr, absolute_sigma=True)
    daParamErr = np.sqrt(np.diag(daPcov))
    daResiduals = daFfdY - fdFfdFit(daFfdX, daPopt[0], daPopt[1])
    dChi2, iDof, dReducedChi2 = fdComputeReducedChi2(
        daResiduals, daFfdYerr, 2)
    return {
        'alpha': daPopt[0], 'alpha_err': daParamErr[0],
        'beta': daPopt[1], 'beta_err': daParamErr[1],
        'popt': daPopt, 'pcov': daPcov,
        'chi2': dChi2, 'dof': iDof, 'reduced_chi2': dReducedChi2,
    }


# ---------------------------------------------------------------------------
# Davenport FFD model
# ---------------------------------------------------------------------------

def fdFlareEquation(tInput, dA1, dA2, dA3, dB1, dB2, dB3):
    """Return log flare rate from the Davenport mass-age FFD model.

    Parameters
    ----------
    tInput : tuple of (daLogEnergy, daLogAge, daMass)
    """
    daLogEnergy, daLogAge, daMass = tInput
    dA = dA1 * daLogAge + dA2 * daMass + dA3
    dB = dB1 * daLogAge + dB2 * daMass + dB3
    return daLogEnergy * dA + dB


def fdaInverseFfd(daEnergy, dAlpha, dBeta):
    """Return flare rate from the Ilin+2020 parameterization."""
    return dBeta / (dAlpha - 1) * (np.array(daEnergy) ** (-dAlpha + 1))


# ---------------------------------------------------------------------------
# TESS data acquisition
# ---------------------------------------------------------------------------

def flistDownloadTessData():
    """Download TESS lightcurve data for GJ 1132."""
    print("Downloading TESS lightcurve data for GJ 1132...")
    return lk.search_lightcurve(
        'GJ 1132', mission='TESS', author='SPOC', exptime=120
    ).download_all()


def fdCalculateTotalExposure(lightcurveCollection):
    """Return total exposure time in days from a lightcurve collection."""
    dTotalExposure = 0.0
    for k in range(len(lightcurveCollection)):
        lightcurveCollection[k].normalize().plot()
        dTotalExposure += len(lightcurveCollection[k]) * 2 / 60 / 24
    return dTotalExposure


# ---------------------------------------------------------------------------
# Flare parameter loading
# ---------------------------------------------------------------------------

def ftLoadFlaresFromJson(sFilePath, dLuminosity):
    """Load labeled flares from identifyFlareCandidates JSON output."""
    with open(sFilePath, 'r') as fileHandle:
        dictSession = json.load(fileHandle)
    listFlares = [
        c for c in dictSession['listCandidates']
        if c.get('sLabel') == 'flare'
    ]
    if len(listFlares) == 0:
        raise ValueError(f"No flares labeled in {sFilePath}")
    listSectors = [c['iSectorIndex'] for c in listFlares]
    daTimeStart = np.array([c['dTimeStart'] for c in listFlares])
    daTimeStop = np.array([c['dTimeStop'] for c in listFlares])
    daPeakSigma = np.array([c['dPeakSigma'] for c in listFlares])
    print(f"Loaded {len(listFlares)} flares from {sFilePath}")
    return listSectors, daTimeStart, daTimeStop, dLuminosity, daPeakSigma


def ftGetFlareParameters(sFlareJsonPath=None):
    """Return flare detection parameters and stellar properties for GJ 1132.

    Loads labeled flares from a JSON file when available, otherwise
    falls back to hard-coded values.
    """
    dLuminosity = 4.77e-3 * const.L_sun.to('erg/s')
    if sFlareJsonPath is None:
        sDefaultPath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'TessFlareLightcurves', 'flare_candidates.json')
        if os.path.isfile(sDefaultPath):
            sFlareJsonPath = sDefaultPath
    if sFlareJsonPath is not None:
        return ftLoadFlaresFromJson(sFlareJsonPath, dLuminosity)
    listSectors = [2, 2, 3]
    daTimeStart = np.array([2284.8817, 2291.2813, 3029.6533])
    daTimeStop = np.array([2284.8905, 2291.3374, 3029.6852])
    return listSectors, daTimeStart, daTimeStop, dLuminosity, None


# ---------------------------------------------------------------------------
# Flare equivalent duration computation
# ---------------------------------------------------------------------------

def fdaComputeFlareEquivDurations(lightcurveCollection, listSectors,
                                  daTimeStart, daTimeStop):
    """Compute equivalent durations for detected flares."""
    print("\nComputing flare equivalent durations...")
    daEquivDuration = np.zeros(len(daTimeStart), dtype=float)
    daTrapezoid = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)
    for k in range(len(daTimeStart)):
        daFlareIdx = np.where(
            (lightcurveCollection[listSectors[k]]['time'].value >= daTimeStart[k])
            & (lightcurveCollection[listSectors[k]]['time'].value <= daTimeStop[k])
        )[0]
        daFlux = lightcurveCollection[listSectors[k]].normalize()['flux'].value[daFlareIdx]
        daTime = lightcurveCollection[listSectors[k]]['time'].value[daFlareIdx]
        daEquivDuration[k] = daTrapezoid(daFlux - 1, daTime * 60 * 60 * 24)
        print(f"  Flare {k}: ED = {daEquivDuration[k]:.2e} s")
    return daEquivDuration


# ---------------------------------------------------------------------------
# FFD computation pipeline
# ---------------------------------------------------------------------------

def ftComputeAndFitFfd(daEquivDuration, dTotalExposure, dLuminosity,
                       lightcurveCollection, daTimeStart, daTimeStop):
    """Compute FFD and fit a power law. Return (x, y, xerr, yerr, dictFit)."""
    print("\nComputing flare frequency distribution...")
    daFfdX, daFfdY, daFfdXerr, daFfdYerr = FFD(
        daEquivDuration, TOTEXP=dTotalExposure,
        Lum=np.log10(dLuminosity.value),
        fluxerr=np.nanmedian(
            lightcurveCollection[0].normalize()['flux_err']),
        dur=daTimeStop - daTimeStart, logY=True)
    print("\nFitting FFD power law...")
    dictFitResults = fdictComputeFfdBestfit(daFfdX, daFfdY, daFfdYerr)
    return daFfdX, daFfdY, daFfdXerr, daFfdYerr, dictFitResults


def fnPrintFfdResults(dictFitResults):
    """Print formatted FFD fitting results."""
    print("\n" + "=" * 60)
    print("GJ 1132 Flare Frequency Distribution Best-Fit Results")
    print("=" * 60)
    print(f"Power-law form: log(Rate) = alpha * log(Energy) + beta")
    print(f"\nAlpha (slope):     {dictFitResults['alpha']:7.4f}"
          f" +/- {dictFitResults['alpha_err']:.4f}")
    print(f"Beta (intercept):  {dictFitResults['beta']:7.4f}"
          f" +/- {dictFitResults['beta_err']:.4f}")
    print(f"\nChi-squared:       {dictFitResults['chi2']:.4f}")
    print(f"Degrees of freedom: {dictFitResults['dof']}")
    print(f"Reduced chi^2:     {dictFitResults['reduced_chi2']:.4f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Literature and cluster data
# ---------------------------------------------------------------------------

def fdictGetLiteratureData():
    """Return GJ 4083 and GJ 1243 FFD data from Hawley+2014."""
    return {
        'gj4083_x': np.array([30.746478873239436, 31.184663536776213]),
        'gj4083_y': np.array([0.03380728065018358, 0.017198670673630925]),
        'gj1243_x': np.array([30.363067292644757, 33.070422535211264]),
        'gj1243_y': np.array([20.605975259769426, 0.03773912345819717]),
    }


def fdictLoadKeplerPosterior(sFilePath):
    """Load Kepler FFD posterior statistics from JSON."""
    with open(sFilePath, "r") as fileHandle:
        dictRaw = json.load(fileHandle)
    dictRaw["daMedians"] = np.array(dictRaw["daMedians"])
    dictRaw["daCovarianceMatrix"] = np.array(dictRaw["daCovarianceMatrix"])
    return dictRaw


def fdaGetDefaultModelParams():
    """Return default Davenport model parameters when no posterior is given."""
    return np.array([-0.148, 0.517, -0.618, 4.69, -16.45, 19.446])


def fdictGetIlinClusterObservations():
    """Return Ilin+2020 cluster observations as a dict."""
    import vplot

    return {
        'cluster': ['Hyades (690Myr)', 'Pleiades (130Myr)',
                     'Praesepe (750Myr)'],
        'ages': np.array([690, 130, 750]),
        'alpha': np.array([1.89, 2.06, 2.00]),
        'beta': np.array([2.3e29, 8.1e34, 8.8e32]),
        'Emin': np.array([1.31e32, 2.32e32, 0.4e33]),
        'Emax': np.array([0.84e34, 2.11e34, 0.36e35]),
        'colors': [vplot.colors.dark_blue, vplot.colors.red,
                   vplot.colors.orange],
    }


def fdictGetClusterData(dictKeplerPosterior=None):
    """Return Ilin+2020 cluster data and Barnes+2026 model parameters."""
    if dictKeplerPosterior is not None:
        daParams = dictKeplerPosterior["daMedians"]
        daCovarianceMatrix = dictKeplerPosterior["daCovarianceMatrix"]
    else:
        daParams = fdaGetDefaultModelParams()
        daCovarianceMatrix = None
    dictCluster = fdictGetIlinClusterObservations()
    dictCluster['params'] = daParams
    dictCluster['covariance'] = daCovarianceMatrix
    return dictCluster


# ---------------------------------------------------------------------------
# Kepler vs TESS statistical comparison
# ---------------------------------------------------------------------------

def fdaComputeProjectionJacobian(dLogAge, dMass):
    """Return the 2x6 Jacobian projecting (a1..b3) to (alpha, beta)."""
    return np.array([
        [dLogAge, dMass, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, dLogAge, dMass, 1.0],
    ])


def fdComputeJointTensionSigma(daDelta, daCovarianceMatrix):
    """Return (chi2, pValue, sigma) for a 2D offset.

    See Press et al. (1992), Numerical Recipes, Section 15.6.
    """
    from scipy.stats import chi2 as chi2dist, norm

    daCovarianceInverse = np.linalg.inv(daCovarianceMatrix)
    dChi2 = float(daDelta @ daCovarianceInverse @ daDelta)
    dPValue = 1.0 - chi2dist.cdf(dChi2, df=2)
    dSigma = float(norm.ppf(1.0 - dPValue / 2.0))
    return dChi2, dPValue, dSigma


def fdictComputeKeplerTessDiscrepancy(dictFitResults, daMedians,
                                      daCovarianceMatrix):
    """Compute Kepler-predicted vs TESS-measured alpha/beta discrepancy.

    Returns a dict with predicted values, deltas, marginal and joint
    tension statistics.
    """
    dLogAge = np.log10(8000)
    dMass = 0.2
    daJacobian = fdaComputeProjectionJacobian(dLogAge, dMass)
    daProjectedCov = daJacobian @ daCovarianceMatrix @ daJacobian.T
    dPredAlpha = daMedians[0] * dLogAge + daMedians[1] * dMass + daMedians[2]
    dPredBeta = daMedians[3] * dLogAge + daMedians[4] * dMass + daMedians[5]
    daDelta = np.array([
        dictFitResults["alpha"] - dPredAlpha,
        dictFitResults["beta"] - dPredBeta])
    daTotalCov = daProjectedCov + dictFitResults["pcov"]
    dChi2, dPValue, dJointSigma = fdComputeJointTensionSigma(
        daDelta, daTotalCov)
    dMargAlpha = abs(daDelta[0]) / np.sqrt(
        daProjectedCov[0, 0] + dictFitResults["alpha_err"] ** 2)
    dMargBeta = abs(daDelta[1]) / np.sqrt(
        daProjectedCov[1, 1] + dictFitResults["beta_err"] ** 2)
    return {
        'dPredAlpha': dPredAlpha, 'dPredBeta': dPredBeta,
        'dMarginalAlpha': dMargAlpha, 'dMarginalBeta': dMargBeta,
        'dChi2': dChi2, 'dPValue': dPValue, 'dJointSigma': dJointSigma,
    }


def fnPrintKeplerTessDiscrepancy(dictFitResults, daMedians,
                                 daCovarianceMatrix):
    """Print Kepler-predicted vs TESS-measured alpha/beta discrepancy."""
    d = fdictComputeKeplerTessDiscrepancy(
        dictFitResults, daMedians, daCovarianceMatrix)
    dCorr = (dictFitResults['pcov'][0, 1]
             / (dictFitResults['alpha_err'] * dictFitResults['beta_err']))
    print("\n" + "=" * 60)
    print("Kepler vs TESS discrepancy")
    print("=" * 60)
    print(f"  Predicted: alpha={d['dPredAlpha']:.4f}, beta={d['dPredBeta']:.4f}")
    print(f"  Observed:  alpha={dictFitResults['alpha']:.4f}, "
          f"beta={dictFitResults['beta']:.4f}")
    print(f"  Marginal: alpha={d['dMarginalAlpha']:.2f}s, "
          f"beta={d['dMarginalBeta']:.2f}s | corr={dCorr:.4f}")
    print(f"  Joint: chi2={d['dChi2']:.2f} (2 DOF), "
          f"p={d['dPValue']:.4f}, sigma={d['dJointSigma']:.2f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Prediction / uncertainty bands
# ---------------------------------------------------------------------------

def fdaComputeTessFitBand(daLogEnergy, dAlpha, dBeta,
                          daCovarianceMatrix, iNumSamples=1000):
    """Return (daLower, daUpper) 1-sigma band for the TESS FFD fit."""
    daSamples = np.random.multivariate_normal(
        [dAlpha, dBeta], daCovarianceMatrix, size=iNumSamples)
    daRates = np.array([
        fdFfdFit(daLogEnergy, s[0], s[1]) for s in daSamples])
    daLower = np.percentile(daRates, 16, axis=0)
    daUpper = np.percentile(daRates, 84, axis=0)
    return daLower, daUpper


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------

def ftRunPipeline(dictKeplerPosterior=None, sFlareJsonPath=None):
    """Run the shared TESS flare analysis pipeline.

    Returns a tuple of all intermediate results needed by any plot.
    """
    lightcurveCollection = flistDownloadTessData()
    dTotalExposure = fdCalculateTotalExposure(lightcurveCollection)
    print(f"Total exposure time: {dTotalExposure:.2f} days")
    print(f"Number of sectors: {len(lightcurveCollection)}")

    (listSectors, daTimeStart, daTimeStop,
     dLuminosity, daPeakSigma) = ftGetFlareParameters(sFlareJsonPath)

    daEquivDuration = fdaComputeFlareEquivDurations(
        lightcurveCollection, listSectors, daTimeStart, daTimeStop)

    (daFfdX, daFfdY, daFfdXerr, daFfdYerr,
     dictFitResults) = ftComputeAndFitFfd(
        daEquivDuration, dTotalExposure, dLuminosity,
        lightcurveCollection, daTimeStart, daTimeStop)
    fnPrintFfdResults(dictFitResults)

    dAlpha = dictFitResults['alpha']
    dBeta = dictFitResults['beta']
    dictLiterature = fdictGetLiteratureData()
    dictCluster = fdictGetClusterData(
        dictKeplerPosterior=dictKeplerPosterior)

    return (lightcurveCollection, listSectors, daTimeStart, daTimeStop,
            daPeakSigma, daFfdX, daFfdY, daFfdXerr, daFfdYerr,
            dictFitResults, dAlpha, dBeta, dictLiterature, dictCluster)


# ---------------------------------------------------------------------------
# Prediction band computation
# ---------------------------------------------------------------------------

def fdaComputePredictionBand(daLogEnergy, daParams,
                             daCovarianceMatrix, dLogAge, dMass,
                             iNumSamples=1000):
    """Return (median, lo1, hi1, lo2, hi2) percentile envelopes.

    Draws samples from the 6-parameter posterior and evaluates
    fdFlareEquation at each.
    """
    daSamples = np.random.multivariate_normal(
        daParams, daCovarianceMatrix, size=iNumSamples)
    daLogAgeArray = np.full_like(daLogEnergy, dLogAge)
    daMassArray = np.full_like(daLogEnergy, dMass)
    daRates = np.array([
        fdFlareEquation(
            (daLogEnergy, daLogAgeArray, daMassArray), *s)
        for s in daSamples
    ])
    daMedian = np.percentile(daRates, 50, axis=0)
    daLower1 = np.percentile(daRates, 16, axis=0)
    daUpper1 = np.percentile(daRates, 84, axis=0)
    daLower2 = np.percentile(daRates, 2.5, axis=0)
    daUpper2 = np.percentile(daRates, 97.5, axis=0)
    return daMedian, daLower1, daUpper1, daLower2, daUpper2


# ---------------------------------------------------------------------------
# Age analysis
# ---------------------------------------------------------------------------

def _fnAppendAgeSample(listAges, dObserved, dCoeff1, dCoeff2,
                       dCoeff3, dStellarMass, dInitialGuess):
    """Solve for age from one MC sample and append if valid."""
    from scipy.optimize import fsolve

    def fnEquation(dLogAge):
        return dObserved - (dCoeff1 * dLogAge
                            + dCoeff2 * dStellarMass + dCoeff3)
    try:
        dLogAge = fsolve(fnEquation, dInitialGuess)[0]
        if 1.0 < dLogAge < 4.2:
            listAges.append(10 ** dLogAge)
    except Exception:
        pass
    return listAges


def _fdictBuildAgeResults(listAlphaMyr, listBetaMyr):
    """Assemble the age results dictionary from MC sample lists."""
    daAlphaMyr = np.array(listAlphaMyr)
    daAlphaGyr = daAlphaMyr / 1000.0
    daLogAlpha = np.log10(daAlphaMyr)

    daBetaMyr = np.array(listBetaMyr)
    daBetaGyr = daBetaMyr / 1000.0
    daLogBeta = np.log10(daBetaMyr)

    return {
        'ages_from_alpha_myr': daAlphaMyr,
        'ages_from_alpha_gyr': daAlphaGyr,
        'log_ages_from_alpha': daLogAlpha,
        'median_age_alpha_gyr': np.median(daAlphaGyr),
        'age_16_alpha_gyr': np.percentile(daAlphaGyr, 16),
        'age_84_alpha_gyr': np.percentile(daAlphaGyr, 84),
        'n_valid_samples_alpha': len(daAlphaMyr),
        'ages_from_beta_myr': daBetaMyr,
        'ages_from_beta_gyr': daBetaGyr,
        'log_ages_from_beta': daLogBeta,
        'median_age_beta_gyr': np.median(daBetaGyr),
        'age_16_beta_gyr': np.percentile(daBetaGyr, 16),
        'age_84_beta_gyr': np.percentile(daBetaGyr, 84),
        'n_valid_samples_beta': len(daBetaMyr),
    }


def fdictComputeAgeFromFfd(dAlpha, dBeta, dictFitResults, daParams,
                           dStellarMass=0.2, dLogEnergy=31.5,
                           iNumSamples=10000):
    """Compute stellar age from FFD parameters via Monte Carlo sampling.

    Uses fdFlareEquation to infer age from observed flare activity.
    """
    daPcov = dictFitResults['pcov']
    daSamples = np.random.multivariate_normal(
        [dAlpha, dBeta], daPcov, size=iNumSamples)

    dA1, dA2, dA3, dB1, dB2, dB3 = daParams

    listAgesFromAlphaMyr = []
    listAgesFromBetaMyr = []
    dInitialGuess = 3.0

    for i in range(iNumSamples):
        listAgesFromAlphaMyr = _fnAppendAgeSample(
            listAgesFromAlphaMyr, daSamples[i, 0],
            dA1, dA2, dA3, dStellarMass, dInitialGuess)
        listAgesFromBetaMyr = _fnAppendAgeSample(
            listAgesFromBetaMyr, daSamples[i, 1],
            dB1, dB2, dB3, dStellarMass, dInitialGuess)

    return _fdictBuildAgeResults(
        listAgesFromAlphaMyr, listAgesFromBetaMyr)


def fdictRunAgeAnalysis(dAlpha, dBeta, dictFitResults, daParams,
                        dStellarMass=0.2):
    """Run age computation from FFD and return results dictionary."""
    print("\n" + "=" * 60)
    print("Computing age of GJ 1132 from flare activity...")
    print("=" * 60)
    print(f"Assumed stellar mass: {dStellarMass} M_sun")
    return fdictComputeAgeFromFfd(
        dAlpha, dBeta, dictFitResults, daParams,
        dStellarMass=dStellarMass, iNumSamples=10000)


def fnPrintAgeResults(dictAgeResults):
    """Print formatted age analysis results."""
    print(f"\nMonte Carlo sampling complete:")
    print(f"  Valid samples from alpha: "
          f"{dictAgeResults['n_valid_samples_alpha']} / 10000")
    print(f"  Valid samples from beta:  "
          f"{dictAgeResults['n_valid_samples_beta']} / 10000")

    dMedianAlpha = dictAgeResults['median_age_alpha_gyr']
    d16Alpha = dictAgeResults['age_16_alpha_gyr']
    d84Alpha = dictAgeResults['age_84_alpha_gyr']
    print(f"\nAge estimate from ALPHA (slope):")
    print(f"  Median age:  {dMedianAlpha:.2f} Gyr")
    print(f"  1-sigma range: {dMedianAlpha:.2f} "
          f"+{d84Alpha - dMedianAlpha:.2f} "
          f"-{dMedianAlpha - d16Alpha:.2f} Gyr")

    dMedianBeta = dictAgeResults['median_age_beta_gyr']
    d16Beta = dictAgeResults['age_16_beta_gyr']
    d84Beta = dictAgeResults['age_84_beta_gyr']
    print(f"\nAge estimate from BETA (intercept):")
    print(f"  Median age:  {dMedianBeta:.2f} Gyr")
    print(f"  1-sigma range: {dMedianBeta:.2f} "
          f"+{d84Beta - dMedianBeta:.2f} "
          f"-{dMedianBeta - d16Beta:.2f} Gyr")
    print("=" * 60 + "\n")


def _fnWriteAgeHeader(fileHandle, dictAgeResults, dStellarMass):
    """Write metadata header for age output file."""
    fileHandle.write(
        "# Age estimates for GJ 1132 from flare frequency "
        "distribution analysis\n")
    fileHandle.write(
        "# Derived using Barnes et al. 2026 fdFlareEquation model\n")
    fileHandle.write(
        f"# Assumed stellar mass: {dStellarMass} M_sun\n")
    fileHandle.write(
        "# Number of Monte Carlo samples: 10000\n#\n")

    for sMethod, sLabel in [('alpha', 'ALPHA (slope)'),
                            ('beta', 'BETA (intercept)')]:
        dMedian = dictAgeResults[f'median_age_{sMethod}_gyr']
        d16 = dictAgeResults[f'age_16_{sMethod}_gyr']
        d84 = dictAgeResults[f'age_84_{sMethod}_gyr']
        iValid = dictAgeResults[f'n_valid_samples_{sMethod}']
        fileHandle.write(
            f"# Summary statistics from {sLabel}:\n")
        fileHandle.write(f"# Valid samples: {iValid}\n")
        fileHandle.write(f"# Median age: {dMedian:.4f} Gyr\n")
        fileHandle.write(f"# 16th percentile: {d16:.4f} Gyr\n")
        fileHandle.write(f"# 84th percentile: {d84:.4f} Gyr\n")
        fileHandle.write(
            f"# 1-sigma upper: +{d84 - dMedian:.4f} Gyr\n")
        fileHandle.write(
            f"# 1-sigma lower: -{dMedian - d16:.4f} Gyr\n#\n")

    fileHandle.write(
        "# Column 1: Age from Alpha (Gyr)\n"
        "# Column 2: Age from Alpha (Myr)\n"
        "# Column 3: log(Age from Alpha in Myr)\n"
        "# Column 4: Age from Beta (Gyr)\n"
        "# Column 5: Age from Beta (Myr)\n"
        "# Column 6: log(Age from Beta in Myr)\n#\n")


def _fnWriteAgeRows(fileHandle, dictAgeResults):
    """Write the data rows for the age output file."""
    daAlphaGyr = dictAgeResults['ages_from_alpha_gyr']
    daAlphaMyr = dictAgeResults['ages_from_alpha_myr']
    daLogAlpha = dictAgeResults['log_ages_from_alpha']
    daBetaGyr = dictAgeResults['ages_from_beta_gyr']
    daBetaMyr = dictAgeResults['ages_from_beta_myr']
    daLogBeta = dictAgeResults['log_ages_from_beta']
    iMaxLength = max(len(daAlphaGyr), len(daBetaGyr))
    for i in range(iMaxLength):
        if i < len(daAlphaGyr):
            fileHandle.write(
                f"{daAlphaGyr[i]:.6f}  "
                f"{daAlphaMyr[i]:.6f}  "
                f"{daLogAlpha[i]:.6f}  ")
        else:
            fileHandle.write("NaN  NaN  NaN  ")
        if i < len(daBetaGyr):
            fileHandle.write(
                f"{daBetaGyr[i]:.6f}  "
                f"{daBetaMyr[i]:.6f}  "
                f"{daLogBeta[i]:.6f}\n")
        else:
            fileHandle.write("NaN  NaN  NaN\n")


def fnSaveAgeData(dictAgeResults, dStellarMass):
    """Save age estimates to text file."""
    sOutputFilename = 'GJ1132_plausible_ages.txt'
    with open(sOutputFilename, 'w') as fileHandle:
        _fnWriteAgeHeader(fileHandle, dictAgeResults, dStellarMass)
        _fnWriteAgeRows(fileHandle, dictAgeResults)
    print(f"Saved plausible ages to: {sOutputFilename}")


# ---------------------------------------------------------------------------
# Panel layout helper
# ---------------------------------------------------------------------------

def ftComputePanelLayout(iNumFlares):
    """Return (iRows, iCols) for the flare lightcurve grid.

    1-3 flares: single row.  4-6 flares: two rows of 2-3 columns.
    """
    if iNumFlares <= 3:
        return 1, iNumFlares
    iCols = (iNumFlares + 1) // 2
    return 2, iCols
