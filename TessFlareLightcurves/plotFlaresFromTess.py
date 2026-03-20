#!/usr/bin/env python
"""
Flare Frequency Distribution Analysis for GJ 1132 using TESS Data

This script downloads TESS lightcurve data for the star GJ 1132, identifies
flares, computes their equivalent durations and energies, and builds a
flare frequency distribution (FFD) for comparison with other M dwarf stars.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import lightkurve as lk
import sys
from scipy.optimize import curve_fit
import vplot

# Add FFD module to path
#sys.path.append('../FFD/')
from FFD import FFD

# Configure matplotlib
matplotlib = plt.matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})


def fdFfdFit(x, alpha, beta):
    """FFD fitting function with both slope and intercept as free parameters"""
    return beta + (x * alpha)


def fdictComputeFfdBestfit(ffd_x, ffd_y, ffd_yerr):
    """
    Compute best-fit FFD parameters and uncertainties for GJ 1132

    Fits a power law of the form: log(Rate) = alpha * log(Energy) + beta
    where both alpha (slope) and beta (intercept) are free parameters.

    Parameters
    ----------
    ffd_x : array-like
        Log flare energies (log10[erg])
    ffd_y : array-like
        Log flare rates (log10[day^-1])
    ffd_yerr : array-like
        Uncertainties in log flare rates

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'alpha': Best-fit power-law slope
        - 'alpha_err': 1-sigma uncertainty in alpha
        - 'beta': Best-fit intercept
        - 'beta_err': 1-sigma uncertainty in beta
        - 'popt': Best-fit parameters from curve_fit [alpha, beta]
        - 'pcov': Covariance matrix from curve_fit
        - 'chi2': Chi-squared statistic
        - 'dof': Degrees of freedom
        - 'reduced_chi2': Reduced chi-squared statistic
    """
    # Perform weighted least-squares fit with initial guess
    p0 = [-1.0, 30.0]  # Initial guess: alpha=-1, beta=30
    popt, pcov = curve_fit(fdFfdFit, ffd_x, ffd_y, p0=p0, sigma=ffd_yerr, absolute_sigma=True)

    # Extract best-fit parameters and uncertainties
    alpha = popt[0]
    beta = popt[1]
    param_err = np.sqrt(np.diag(pcov))
    alpha_err = param_err[0]
    beta_err = param_err[1]

    # Compute reduced chi-squared
    residuals = ffd_y - fdFfdFit(ffd_x, alpha, beta)
    chi2 = np.sum((residuals / ffd_yerr)**2)
    dof = len(ffd_x) - 2  # Number of data points minus number of free parameters
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan

    results = {
        'alpha': alpha,
        'alpha_err': alpha_err,
        'beta': beta,
        'beta_err': beta_err,
        'popt': popt,
        'pcov': pcov,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2
    }

    return results


def fdFlareEquation(X, a1, a2, a3, b1, b2, b3):
    """
    FFD fitting equation with powerlaw slope and intercept as functions of mass and age

    Parameters
    ----------
    X : tuple of (logE, logt, m)
        logE : log flare energy in erg
        logt : log age in Myr
        m : stellar mass in Solar masses

    Returns
    -------
    logR : log rate of flares per day
    """
    logE, logt, m = X

    a = a1 * logt + a2 * m + a3
    b = b1 * logt + b2 * m + b3
    logR = logE * a + b

    return logR


def fdaInverseFfd(E, alpha, beta):
    """
    Compute flare frequency from Ilin+2020 parameterization

    Parameters
    ----------
    E : array-like
        Flare energies in erg
    alpha : float
        Power law slope
    beta : float
        Normalization constant

    Returns
    -------
    f : array-like
        Flare rate
    """
    f = beta / (alpha - 1) * (E**(-alpha + 1))
    return f


def fdictComputeAgeFromFfd(alpha, beta, fit_results, params, stellar_mass=0.2,
                         log_energy=31.5, n_samples=10000):
    """
    Compute stellar age from FFD parameters using Monte Carlo sampling

    Uses the fdFlareEquation model to infer age from observed flare activity.
    The model relates alpha (slope) and beta (intercept) to stellar age and mass.
    Solves for age using both alpha and beta equations independently.

    Parameters
    ----------
    alpha : float
        Best-fit FFD slope
    beta : float
        Best-fit FFD intercept
    fit_results : dict
        Dictionary containing alpha_err, beta_err, and pcov
    params : array-like
        fdFlareEquation model parameters [a1, a2, a3, b1, b2, b3]
    stellar_mass : float, optional
        Stellar mass in solar masses (default: 0.2)
    log_energy : float, optional
        Log energy at which to evaluate (default: 31.5)
    n_samples : int, optional
        Number of Monte Carlo samples (default: 10000)

    Returns
    -------
    age_results : dict
        Dictionary containing:
        - 'ages_from_alpha_myr': Array of ages from alpha in Myr
        - 'ages_from_alpha_gyr': Array of ages from alpha in Gyr
        - 'ages_from_beta_myr': Array of ages from beta in Myr
        - 'ages_from_beta_gyr': Array of ages from beta in Gyr
        - 'log_ages_from_alpha': Array of log(age) from alpha in log(Myr)
        - 'log_ages_from_beta': Array of log(age) from beta in log(Myr)
        - Statistics for both methods
    """
    from scipy.optimize import fsolve

    # Extract covariance matrix for alpha and beta
    pcov = fit_results['pcov']

    # Generate correlated samples of alpha and beta using covariance matrix
    samples = np.random.multivariate_normal([alpha, beta], pcov, size=n_samples)
    alpha_samples = samples[:, 0]
    beta_samples = samples[:, 1]

    # Unpack fdFlareEquation parameters
    a1, a2, a3, b1, b2, b3 = params

    ages_from_alpha_myr = []
    ages_from_beta_myr = []

    for i in range(n_samples):
        alpha_i = alpha_samples[i]
        beta_i = beta_samples[i]

        # From fdFlareEquation:
        # alpha = a1 * logt + a2 * m + a3
        # beta = b1 * logt + b2 * m + b3
        #
        # Solve using alpha
        def eq_alpha(logt):
            return alpha_i - (a1 * logt + a2 * stellar_mass + a3)

        # Solve using beta
        def eq_beta(logt):
            return beta_i - (b1 * logt + b2 * stellar_mass + b3)

        # Initial guess: 3.0 (log Myr) = 1000 Myr = 1 Gyr
        initial_guess = 3.0

        # Solve for age using alpha
        try:
            logt_from_alpha = fsolve(eq_alpha, initial_guess)[0]
            # Only accept solutions that give reasonable ages (10 Myr to 15 Gyr)
            if 1.0 < logt_from_alpha < 4.2:
                ages_from_alpha_myr.append(10**logt_from_alpha)
        except:
            pass

        # Solve for age using beta
        try:
            logt_from_beta = fsolve(eq_beta, initial_guess)[0]
            # Only accept solutions that give reasonable ages (10 Myr to 15 Gyr)
            if 1.0 < logt_from_beta < 4.2:
                ages_from_beta_myr.append(10**logt_from_beta)
        except:
            pass

    ages_from_alpha_myr = np.array(ages_from_alpha_myr)
    ages_from_alpha_gyr = ages_from_alpha_myr / 1000.0
    log_ages_from_alpha = np.log10(ages_from_alpha_myr)

    ages_from_beta_myr = np.array(ages_from_beta_myr)
    ages_from_beta_gyr = ages_from_beta_myr / 1000.0
    log_ages_from_beta = np.log10(ages_from_beta_myr)

    # Compute statistics for alpha-derived ages
    median_age_alpha_gyr = np.median(ages_from_alpha_gyr)
    age_16_alpha_gyr = np.percentile(ages_from_alpha_gyr, 16)
    age_84_alpha_gyr = np.percentile(ages_from_alpha_gyr, 84)

    # Compute statistics for beta-derived ages
    median_age_beta_gyr = np.median(ages_from_beta_gyr)
    age_16_beta_gyr = np.percentile(ages_from_beta_gyr, 16)
    age_84_beta_gyr = np.percentile(ages_from_beta_gyr, 84)

    age_results = {
        'ages_from_alpha_myr': ages_from_alpha_myr,
        'ages_from_alpha_gyr': ages_from_alpha_gyr,
        'log_ages_from_alpha': log_ages_from_alpha,
        'median_age_alpha_gyr': median_age_alpha_gyr,
        'age_16_alpha_gyr': age_16_alpha_gyr,
        'age_84_alpha_gyr': age_84_alpha_gyr,
        'n_valid_samples_alpha': len(ages_from_alpha_myr),

        'ages_from_beta_myr': ages_from_beta_myr,
        'ages_from_beta_gyr': ages_from_beta_gyr,
        'log_ages_from_beta': log_ages_from_beta,
        'median_age_beta_gyr': median_age_beta_gyr,
        'age_16_beta_gyr': age_16_beta_gyr,
        'age_84_beta_gyr': age_84_beta_gyr,
        'n_valid_samples_beta': len(ages_from_beta_myr)
    }

    return age_results


# ===== Data Acquisition Functions =====

def flistDownloadTessData():
    """Download TESS lightcurve data for GJ 1132"""
    print("Downloading TESS lightcurve data for GJ 1132...")
    return lk.search_lightcurve('GJ 1132', mission='TESS', author='SPOC', exptime=120).download_all()


def fdCalculateTotalExposure(lc):
    """Calculate total exposure time from lightcurve collection"""
    totexp = 0
    for k in range(len(lc)):
        lc[k].normalize().plot()
        totexp += (len(lc[k]) * 2/60/24)  # 2 min exposures
    return totexp


def ftGetFlareParameters(sFlareJsonPath=None):
    """Return flare detection parameters and stellar properties for GJ 1132.

    If sFlareJsonPath is provided, load labeled flares from that JSON.
    Otherwise, look for flare_candidates.json next to this script.
    Falls back to hard-coded values only if no JSON file is found.
    """
    import os
    dLuminosity = 4.77e-3 * const.L_sun.to('erg/s')
    if sFlareJsonPath is None:
        sDefaultPath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'flare_candidates.json')
        if os.path.isfile(sDefaultPath):
            sFlareJsonPath = sDefaultPath
    if sFlareJsonPath is not None:
        return ftLoadFlaresFromJson(sFlareJsonPath, dLuminosity)
    sectors = [2, 2, 3]
    t_start = np.array([2284.8817, 2291.2813, 3029.6533])
    t_stop = np.array([2284.8905, 2291.3374, 3029.6852])
    return sectors, t_start, t_stop, dLuminosity, None


def ftLoadFlaresFromJson(sFilePath, dLuminosity):
    """Load labeled flares from identifyFlareCandidates.py JSON output."""
    with open(sFilePath, 'r') as fileHandle:
        dictSession = json.load(fileHandle)
    listFlares = [c for c in dictSession['listCandidates']
                  if c.get('sLabel') == 'flare']
    if len(listFlares) == 0:
        raise ValueError(f"No flares labeled in {sFilePath}")
    sectors = [c['iSectorIndex'] for c in listFlares]
    t_start = np.array([c['dTimeStart'] for c in listFlares])
    t_stop = np.array([c['dTimeStop'] for c in listFlares])
    daPeakSigma = np.array([c['dPeakSigma'] for c in listFlares])
    print(f"Loaded {len(listFlares)} flares from {sFilePath}")
    return sectors, t_start, t_stop, dLuminosity, daPeakSigma


# ===== Literature Data Functions =====

def fdictGetLiteratureData():
    """Return GJ 4083 and GJ 1243 FFD data from Hawley+2014"""
    gj4083_x = np.array([30.746478873239436, 31.184663536776213])
    gj4083_y = np.array([0.03380728065018358, 0.017198670673630925])
    gj1243_x = np.array([30.363067292644757, 33.070422535211264])
    gj1243_y = np.array([20.605975259769426, 0.03773912345819717])
    return {'gj4083_x': gj4083_x, 'gj4083_y': gj4083_y,
            'gj1243_x': gj1243_x, 'gj1243_y': gj1243_y}


def fdictLoadKeplerPosterior(sFilePath):
    """Load Kepler FFD posterior statistics from JSON."""
    with open(sFilePath, "r") as fileHandle:
        dictRaw = json.load(fileHandle)
    dictRaw["daMedians"] = np.array(dictRaw["daMedians"])
    dictRaw["daCovarianceMatrix"] = np.array(dictRaw["daCovarianceMatrix"])
    return dictRaw


def fdaComputeProjectionJacobian(dLogAge, dMass):
    """Return the 2x6 Jacobian projecting (a1..b3) to (alpha, beta).

    The Davenport model is linear in the six parameters:
      alpha = a1*logAge + a2*mass + a3
      beta  = b1*logAge + b2*mass + b3
    so the Jacobian is exact (no approximation).
    """
    daJacobian = np.array([
        [dLogAge, dMass, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, dLogAge, dMass, 1.0],
    ])
    return daJacobian


def fdComputeJointTensionSigma(daDelta, daCovarianceMatrix):
    """Return the equivalent sigma for a 2D offset given the joint covariance.

    Computes chi2 = delta^T @ C^{-1} @ delta, then converts to an
    equivalent Gaussian sigma via the chi-squared CDF with 2 DOF.
    See Press et al. (1992), Numerical Recipes, Section 15.6.
    """
    from scipy.stats import chi2 as chi2dist
    daCovarianceInverse = np.linalg.inv(daCovarianceMatrix)
    dChi2 = float(daDelta @ daCovarianceInverse @ daDelta)
    dPValue = 1.0 - chi2dist.cdf(dChi2, df=2)
    from scipy.stats import norm
    dSigma = float(norm.ppf(1.0 - dPValue / 2.0))
    return dChi2, dPValue, dSigma


def fnPrintKeplerTessDiscrepancy(dictFitResults, daMedians, daCovarianceMatrix):
    """Quantify Kepler-predicted vs TESS-measured alpha/beta discrepancy.

    Uses the joint 2D chi-squared statistic to account for the strong
    correlation between alpha and beta in the TESS fit.  Marginal
    (1D) tensions are also printed for reference.
    """
    dLogAge = np.log10(8000)  # 8 Gyr in Myr
    dMass = 0.2

    daJacobian = fdaComputeProjectionJacobian(dLogAge, dMass)
    daProjectedCovariance = daJacobian @ daCovarianceMatrix @ daJacobian.T

    dPredAlpha = (daMedians[0] * dLogAge + daMedians[1] * dMass
                  + daMedians[2])
    dPredBeta = (daMedians[3] * dLogAge + daMedians[4] * dMass
                 + daMedians[5])

    daDelta = np.array([
        dictFitResults["alpha"] - dPredAlpha,
        dictFitResults["beta"] - dPredBeta,
    ])
    daTotalCovariance = daProjectedCovariance + dictFitResults["pcov"]
    dChi2, dPValue, dJointSigma = fdComputeJointTensionSigma(
        daDelta, daTotalCovariance)

    dMarginalAlpha = (abs(daDelta[0])
                      / np.sqrt(daProjectedCovariance[0, 0]
                                + dictFitResults["alpha_err"] ** 2))
    dMarginalBeta = (abs(daDelta[1])
                     / np.sqrt(daProjectedCovariance[1, 1]
                               + dictFitResults["beta_err"] ** 2))

    print("\n" + "=" * 60)
    print("Kepler vs TESS discrepancy")
    print("=" * 60)
    print(f"  Predicted (8 Gyr): alpha={dPredAlpha:.4f}, "
          f"beta={dPredBeta:.4f}")
    print(f"  Observed (TESS):   alpha={dictFitResults['alpha']:.4f}, "
          f"beta={dictFitResults['beta']:.4f}")
    print(f"\n  Marginal tensions (ignoring alpha-beta correlation):")
    print(f"    alpha: {dMarginalAlpha:.2f} sigma")
    print(f"    beta:  {dMarginalBeta:.2f} sigma")
    print(f"\n  TESS fit correlation coefficient: "
          f"{dictFitResults['pcov'][0,1] / (dictFitResults['alpha_err'] * dictFitResults['beta_err']):.4f}")
    print(f"\n  Joint 2D tension (accounts for alpha-beta correlation):")
    print(f"    chi2 = {dChi2:.2f} (2 DOF)")
    print(f"    p-value = {dPValue:.4f}")
    print(f"    equivalent sigma = {dJointSigma:.2f}")
    print("=" * 60 + "\n")


def fdictGetClusterData(dictKeplerPosterior=None):
    """Return Ilin+2020 cluster data and Barnes+2026 model parameters.

    If dictKeplerPosterior is provided, use its medians instead of
    hard-coded values.
    """
    if dictKeplerPosterior is not None:
        params = dictKeplerPosterior["daMedians"]
    else:
        params = np.array([-0.148, 0.517, -0.618, 4.69, -16.45, 19.446])
    cluster = ['Hyades (690Myr)', 'Pleiades (130Myr)', 'Praesepe (750Myr)']
    Iages = np.array([690, 130, 750])
    Ialpha = np.array([1.89, 2.06, 2.00])
    Ibeta = np.array([2.3e29, 8.1e34, 8.8e32])
    IEmin = np.array([1.31e32, 2.32e32, 0.4e33])
    IEmax = np.array([0.84e34, 2.11e34, 0.36e35])
    clrs = [vplot.colors.dark_blue, vplot.colors.red, vplot.colors.orange]
    daCovarianceMatrix = None
    if dictKeplerPosterior is not None:
        daCovarianceMatrix = dictKeplerPosterior["daCovarianceMatrix"]
    return {'params': params, 'covariance': daCovarianceMatrix,
            'cluster': cluster, 'ages': Iages,
            'alpha': Ialpha, 'beta': Ibeta, 'Emin': IEmin, 'Emax': IEmax,
            'colors': clrs}


# ===== Flare Analysis Functions =====

def fdaComputeFlareEquivalentDurations(lc, sectors, t_start, t_stop):
    """Compute equivalent durations for detected flares"""
    print("\nComputing flare equivalent durations...")
    ed = np.zeros(len(t_start), dtype=float)
    for k in range(len(t_start)):
        flare = np.where((lc[sectors[k]]['time'].value >= t_start[k]) &
                         (lc[sectors[k]]['time'].value <= t_stop[k]))[0]
        daTrapezoid = getattr(np, 'trapezoid', np.trapz)
        ed[k] = daTrapezoid(lc[sectors[k]].normalize()['flux'].value[flare] - 1,
                            lc[sectors[k]]['time'].value[flare]*60*60*24)
        print(f"  Flare {k}: ED = {ed[k]:.2e} s")
    return ed


def ftComputePanelLayout(iNumFlares):
    """Return (iRows, iCols) for the flare lightcurve grid.

    1-3 flares: single row.  4-6 flares: two rows of 2-3 columns.
    """
    if iNumFlares <= 3:
        return 1, iNumFlares
    iCols = (iNumFlares + 1) // 2
    return 2, iCols


def fnPlotFlareLightcurves(lc, sectors, t_start, t_stop,
                           daPeakSigma=None, sOutputPath=None):
    """Create multi-panel flare lightcurve plot."""
    iNumFlares = len(t_start)
    iRows, iCols = ftComputePanelLayout(iNumFlares)
    fig, axes = plt.subplots(iRows, iCols,
                             figsize=(5 * iCols, 5 * iRows), sharey=True)
    daAxesFlat = np.atleast_1d(axes).flatten()
    for k in range(iNumFlares):
        flare = np.where((lc[sectors[k]]['time'].value >= t_start[k]) &
                         (lc[sectors[k]]['time'].value <= t_stop[k]))[0]
        ax = daAxesFlat[k]
        ax.plot(lc[sectors[k]]['time'].value - np.nanmin(lc[sectors[k]]['time'].value[flare]),
                lc[sectors[k]].normalize()['flux'].value, c='k')
        ax.scatter(lc[sectors[k]]['time'].value[flare] - np.nanmin(lc[sectors[k]]['time'].value[flare]),
                   lc[sectors[k]].normalize()['flux'].value[flare], c=vplot.colors.pale_blue)
        ax.set_xlim(-0.05, 0.1)
        ax.set_xlabel('Time [days]', fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        if k % iCols == 0:
            ax.set_ylabel('Relative Flux', fontsize=20)
        if daPeakSigma is not None:
            ax.text(0.05, 0.95, f'{daPeakSigma[k]:.1f}σ',
                    transform=ax.transAxes, fontsize=16,
                    verticalalignment='top')
    for k in range(iNumFlares, iRows * iCols):
        daAxesFlat[k].set_visible(False)
    plt.tight_layout()
    sFile = sOutputPath if sOutputPath else 'GJ1132_flares.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()


# ===== FFD Analysis Functions =====

def ftComputeAndFitFfd(ed, totexp, lumin, lc, t_start, t_stop):
    """Compute FFD and fit power law"""
    print("\nComputing flare frequency distribution...")
    ffd_x, ffd_y, ffd_xerr, ffd_yerr = FFD(ed, TOTEXP=totexp,
                                            Lum=np.log10(lumin.value),
                                            fluxerr=np.nanmedian(lc[0].normalize()['flux_err']),
                                            dur=t_stop - t_start, logY=True)
    print("\nFitting FFD power law...")
    fit_results = fdictComputeFfdBestfit(ffd_x, ffd_y, ffd_yerr)
    return ffd_x, ffd_y, ffd_xerr, ffd_yerr, fit_results


def fnPrintFfdResults(fit_results):
    """Print formatted FFD fitting results"""
    print("\n" + "="*60)
    print("GJ 1132 Flare Frequency Distribution Best-Fit Results")
    print("="*60)
    print(f"Power-law form: log(Rate) = alpha * log(Energy) + beta")
    print(f"\nAlpha (slope):     {fit_results['alpha']:7.4f} ± {fit_results['alpha_err']:.4f}")
    print(f"Beta (intercept):  {fit_results['beta']:7.4f} ± {fit_results['beta_err']:.4f}")
    print(f"\nChi-squared:       {fit_results['chi2']:.4f}")
    print(f"Degrees of freedom: {fit_results['dof']}")
    print(f"Reduced chi^2:     {fit_results['reduced_chi2']:.4f}")
    print("="*60 + "\n")


# ===== Plotting Functions =====

def fnPlotBasicFfd(ffd_x, ffd_y, ffd_yerr, alpha, beta):
    """Plot basic FFD with power-law fit"""
    plt.figure()
    plt.errorbar(ffd_x, ffd_y, yerr=ffd_yerr, linestyle='none', color='k')
    plt.scatter(ffd_x, ffd_y, c='k')
    plt.plot([31.5, 32.5], fdFfdFit(np.array([31.5, 32.5]), alpha, beta),
             label=r'$\alpha$=' + f'{alpha:.2f}' + r', $\beta$=' + f'{beta:.2f}')
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('log Flare Rate (day$^{-1}$)')
    plt.legend(fontsize=14, loc='lower left')
    plt.savefig('GJ1132_FFD.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')


def fnPlotFfdComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta, lit_data):
    """Plot FFD comparison with GJ 4083 and GJ 1243"""
    plt.figure()
    plt.errorbar(ffd_x, ffd_y, yerr=ffd_yerr, linestyle='none', color='k')
    plt.scatter(ffd_x, ffd_y, c='k')
    plt.plot([31.5, 32.5], fdFfdFit(np.array([31.5, 32.5]), alpha, beta),
             label=r'GJ 1132: $\alpha$=' + f'{alpha:.2f}' + r', $\beta$=' + f'{beta:.2f}')
    plt.plot(lit_data['gj4083_x'], np.log10(lit_data['gj4083_y']), marker='s',
             linestyle='--', label='GJ 4083 (M3, Hawley+2014)')
    plt.plot(lit_data['gj1243_x'], np.log10(lit_data['gj1243_y']),
             c='limegreen', lw=2, label='GJ 1243 (M4, Hawley+2014)')
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('log Flare Rate (day$^{-1}$)')
    plt.legend(fontsize=12)
    plt.savefig('GJ1132_FFD_comp.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')


def fnPlotIlinReproduction(cluster_data):
    """Reproduce Ilin+2020 Fig 5b"""
    params = cluster_data['params']
    plt.figure()
    for k in [1, 0, 2]:
        plt.plot([cluster_data['Emin'][k], cluster_data['Emax'][k]],
                 fdaInverseFfd([cluster_data['Emin'][k], cluster_data['Emax'][k]],
                      cluster_data['alpha'][k], cluster_data['beta'][k]),
                 c=cluster_data['colors'][k], label=cluster_data['cluster'][k])
        X = (np.log10([cluster_data['Emin'][k], cluster_data['Emax'][k]]),
             np.log10([cluster_data['ages'][k], cluster_data['ages'][k]]),
             np.array([0.2, 0.2]))
        plt.plot([cluster_data['Emin'][k], cluster_data['Emax'][k]],
                 10**fdFlareEquation(X, *params)*365.25, c=cluster_data['colors'][k], linestyle='--')
    plt.plot([], c='k', linestyle='--', label='Davenport+2019 model')
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Flare Energy (erg)')
    plt.ylabel('Flare Rate (year$^{-1}$)')


def fdaComputePredictionBand(daLogEnergy, daParams, daCovarianceMatrix,
                             dLogAge, dMass, iNumSamples=1000):
    """Return (daMedian, daLower1, daUpper1, daLower2, daUpper2) for the
    Kepler-predicted FFD evaluated over daLogEnergy.

    Draws samples from the 6-parameter posterior, evaluates fdFlareEquation
    at each, and returns the median plus 1-sigma and 2-sigma percentile
    envelopes.
    """
    daSamples = np.random.multivariate_normal(
        daParams, daCovarianceMatrix, size=iNumSamples)
    daLogAge = np.full_like(daLogEnergy, dLogAge)
    daMass = np.full_like(daLogEnergy, dMass)
    daRates = np.array([
        fdFlareEquation((daLogEnergy, daLogAge, daMass), *s)
        for s in daSamples
    ])
    daMedian = np.percentile(daRates, 50, axis=0)
    daLower1 = np.percentile(daRates, 16, axis=0)
    daUpper1 = np.percentile(daRates, 84, axis=0)
    daLower2 = np.percentile(daRates, 2.5, axis=0)
    daUpper2 = np.percentile(daRates, 97.5, axis=0)
    return daMedian, daLower1, daUpper1, daLower2, daUpper2


def fdaComputeTessFitBand(daLogEnergy, dAlpha, dBeta, daCovarianceMatrix,
                          iNumSamples=1000):
    """Return (daLower, daUpper) 1-sigma band for the TESS FFD fit.

    Draws correlated (alpha, beta) samples from the fit covariance and
    evaluates the linear FFD model at each, returning the 16th and 84th
    percentile envelopes.
    """
    daSamples = np.random.multivariate_normal(
        [dAlpha, dBeta], daCovarianceMatrix, size=iNumSamples)
    daRates = np.array([
        fdFfdFit(daLogEnergy, s[0], s[1]) for s in daSamples
    ])
    daLower = np.percentile(daRates, 16, axis=0)
    daUpper = np.percentile(daRates, 84, axis=0)
    return daLower, daUpper


def fnPlotComprehensiveComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta,
                                  lit_data, cluster_data, fit_results=None,
                                  sOutputPath=None):
    """Create comprehensive FFD comparison plot with clusters."""
    params = cluster_data['params']
    plt.figure()

    iNumPoints = 100
    daLogEnergy = np.linspace(30.0, 34.5, iNumPoints)
    dEnergyMin = np.min(ffd_x)
    dEnergyMax = np.max(ffd_x)
    daLogEnergyData = np.linspace(dEnergyMin, dEnergyMax, iNumPoints)

    if fit_results is not None:
        daLo, daHi = fdaComputeTessFitBand(
            daLogEnergyData, alpha, beta, fit_results['pcov'])
        plt.fill_between(daLogEnergyData, daLo, daHi,
                         color='k', alpha=0.15,
                         label=r'TESS fit 1$\sigma$')

    plt.errorbar(ffd_x, ffd_y, yerr=ffd_yerr, linestyle='none', color='k')
    plt.scatter(ffd_x, ffd_y, c='k')
    plt.plot(daLogEnergyData, fdFfdFit(daLogEnergyData, alpha, beta), c='k')
    plt.text(31.1, -2.7, 'GJ 1132\n(observed)', color='k', fontsize=10)

    plt.plot(lit_data['gj4083_x'], np.log10(lit_data['gj4083_y']), marker='s',
             linestyle='dotted', lw=2, c=vplot.colors.pale_blue)
    plt.text(30.5, -2, 'GJ 4083', color=vplot.colors.pale_blue, fontsize=10)

    plt.plot(lit_data['gj1243_x'], np.log10(lit_data['gj1243_y']),
             c=vplot.colors.pale_blue, linestyle='dotted', lw=2)
    plt.text(31.5, 0.2, 'GJ 1243', color=vplot.colors.pale_blue, fontsize=10)

    for k in [1, 0, 2]:
        plt.plot(np.log10([cluster_data['Emin'][k], cluster_data['Emax'][k]]),
                 np.log10(fdaInverseFfd([cluster_data['Emin'][k], cluster_data['Emax'][k]],
                               cluster_data['alpha'][k], cluster_data['beta'][k])) - np.log10(365.25),
                 c=cluster_data['colors'][k], label=cluster_data['cluster'][k])
        X = (np.log10([cluster_data['Emin'][k], cluster_data['Emax'][k]]),
             np.log10([cluster_data['ages'][k], cluster_data['ages'][k]]),
             np.array([0.2, 0.2]))
        plt.plot(np.log10([cluster_data['Emin'][k], cluster_data['Emax'][k]]),
                 fdFlareEquation(X, *params), c=cluster_data['colors'][k], linestyle='--')

    plt.plot([], c='k', linestyle='--', label='Fit at Cluster Ages')
    dLogAge = np.log10(8000)
    dMass = 0.2
    X = (daLogEnergy, np.full(iNumPoints, dLogAge),
         np.full(iNumPoints, dMass))
    plt.plot(daLogEnergy, fdFlareEquation(X, *params),
             color=vplot.colors.purple, label='Kepler prediction (8 Gyr)')
    plt.text(31.4, -1, 'GJ 1132\n(predicted)',
             color=vplot.colors.purple, fontsize=10)

    plt.xlabel('log Flare Energy [erg]')
    plt.ylabel('log Flare Rate [day$^{-1}$]')
    plt.legend(fontsize=10)
    sFile = sOutputPath if sOutputPath else 'GJ1132_FFD_comp2.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()


def fnPlotAlphaBetaComparison(alpha, beta, fit_results, sOutputPath=None,
                               dictKeplerPosterior=None):
    """Plot alpha-beta parameter space comparison: Kepler vs GJ 1132"""
    print("\n" + "="*60)
    print("Generating alpha-beta parameter space comparison...")
    print("="*60)

    if dictKeplerPosterior is not None:
        kepler_means = dictKeplerPosterior["daMedians"]
        kepler_cov = dictKeplerPosterior["daCovarianceMatrix"]
    else:
        kepler_means = np.array([-0.148, 0.517, -0.618, 4.69, -16.45, 19.446])
        kepler_uncertainties = np.array([0.000529, 0.00244, 0.00231, 0.0180, 0.0824, 0.0779])
        kepler_cov = np.diag(kepler_uncertainties**2)

    iNumDraws = 100
    kepler_samples = np.random.multivariate_normal(kepler_means, kepler_cov,
                                                   size=iNumDraws)

    logt_kepler = np.log10(8000)  # 8 Gyr in Myr
    mass = 0.2
    kepler_alphas = kepler_samples[:, 0] * logt_kepler + kepler_samples[:, 1] * mass + kepler_samples[:, 2]
    kepler_betas = kepler_samples[:, 3] * logt_kepler + kepler_samples[:, 4] * mass + kepler_samples[:, 5]

    gj1132_samples = np.random.multivariate_normal([alpha, beta],
                                                   fit_results['pcov'],
                                                   size=iNumDraws)

    plt.figure(figsize=(10, 8))
    plt.scatter(kepler_alphas, kepler_betas, c=vplot.colors.orange, alpha=0.6, s=50,
                label=r'$Kepler$ (8 Gyr)', edgecolors='k', linewidths=0.5)
    plt.scatter(gj1132_samples[:, 0], gj1132_samples[:, 1], c=vplot.colors.pale_blue,
                alpha=0.6, s=50, label=r'GJ 1132 ($TESS$)', edgecolors='k', linewidths=0.5)
    plt.scatter([-0.32],[8.54],label='Best fit',marker='^',color='red',s=100)
    plt.xlabel(r'Slope ($a$)', fontsize=18)
    plt.ylabel(r'Intercept ($b$)', fontsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.tick_params(axis='both', labelsize=14)
    sFile = sOutputPath if sOutputPath else 'FitComparison.pdf'
    plt.savefig(sFile, dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')
    plt.close()

    print(f"Kepler prediction (8 Gyr): alpha = {np.mean(kepler_alphas):.4f} +/- {np.std(kepler_alphas):.4f}, "
          f"beta = {np.mean(kepler_betas):.4f} +/- {np.std(kepler_betas):.4f}")
    print(f"GJ 1132 measurement: alpha = {alpha:.4f} +/- {fit_results['alpha_err']:.4f}, "
          f"beta = {beta:.4f} +/- {fit_results['beta_err']:.4f}")

    if dictKeplerPosterior is not None:
        fnPrintKeplerTessDiscrepancy(
            fit_results, kepler_means, kepler_cov)

    print("="*60 + "\n")


def fnPlotProximaComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta, lit_data):
    """Plot FFD comparison with Proxima Centauri"""
    plt.figure()
    plt.errorbar(ffd_x, ffd_y, yerr=ffd_yerr, linestyle='none', color='k')
    plt.scatter(ffd_x, ffd_y, c='k')
    plt.plot([31.5, 32.5], fdFfdFit(np.array([31.5, 32.5]), alpha, beta), c='k')
    plt.text(32.5, -2, 'GJ 1132', color='k', fontsize=10)

    plt.plot(lit_data['gj4083_x'], np.log10(lit_data['gj4083_y']),
             marker='s', linestyle='--', c='C1')
    plt.text(30.5, -2, 'GJ 4083', color='C1', fontsize=10)

    plt.plot(lit_data['gj1243_x'], np.log10(lit_data['gj1243_y']),
             c='ForestGreen', lw=3)
    plt.text(31.55, 0.2, 'GJ 1243', color='ForestGreen', fontsize=10)

    plt.plot([30.5, 32.5], -1.22*np.array([30.5, 32.5])+38.1, c='Magenta', lw=3)
    plt.text(30.55, 0, 'Proxima', color='Magenta', fontsize=10)

    plt.fill_between([31.3, 31.4], [-3, -3], [1.3, 1.3], alpha=0.5)
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('log Flare Rate (day$^{-1}$)')


def fnPlotAgeActivityRelation(alpha, beta, lit_data):
    """Plot age-activity relationship"""
    m4083 = float(np.diff(np.log10(lit_data['gj4083_y'])) / np.diff(lit_data['gj4083_x']))
    b4083 = float((np.log10(lit_data['gj4083_y'])[0] - (m4083*lit_data['gj4083_x'][0])))

    m1243 = float((np.log10(lit_data['gj1243_y'][1]) - np.log10(lit_data['gj1243_y'][0])) /
                  (lit_data['gj1243_x'][1] - lit_data['gj1243_x'][0]))
    b1243 = float(np.log10(lit_data['gj1243_y'][1]) - (m1243*lit_data['gj1243_x'][1]))

    ages = np.array([50e6, 4.853e9, 8e9, 10e9])
    f313 = np.array([m1243*31.3 + b1243, -1.22*31.3+38.1,
                     float(fdFfdFit(np.array([31.3]), alpha, beta)), m4083*31.3 + b4083])

    plt.figure()
    plt.scatter(ages, f313)
    plt.xscale('log')
    plt.xlabel('Age (years)')
    plt.ylabel('log Flare Rate at log E=31.3')


# ===== Age Analysis Functions =====

def fdictRunAgeAnalysis(alpha, beta, fit_results, params, stellar_mass=0.2):
    """Run age computation from FFD"""
    print("\n" + "="*60)
    print("Computing age of GJ 1132 from flare activity...")
    print("="*60)
    print(f"Assumed stellar mass: {stellar_mass} M_sun")

    age_results = fdictComputeAgeFromFfd(alpha, beta, fit_results, params,
                                        stellar_mass=stellar_mass, n_samples=10000)
    return age_results


def fnPrintAgeResults(age_results):
    """Print formatted age analysis results"""
    print(f"\nMonte Carlo sampling complete:")
    print(f"  Valid samples from alpha: {age_results['n_valid_samples_alpha']} / 10000")
    print(f"  Valid samples from beta:  {age_results['n_valid_samples_beta']} / 10000")

    print(f"\nAge estimate from ALPHA (slope):")
    print(f"  Median age:  {age_results['median_age_alpha_gyr']:.2f} Gyr")
    print(f"  16th percentile: {age_results['age_16_alpha_gyr']:.2f} Gyr")
    print(f"  84th percentile: {age_results['age_84_alpha_gyr']:.2f} Gyr")
    print(f"  1-sigma range: {age_results['median_age_alpha_gyr']:.2f} "
          f"+{age_results['age_84_alpha_gyr']-age_results['median_age_alpha_gyr']:.2f} "
          f"-{age_results['median_age_alpha_gyr']-age_results['age_16_alpha_gyr']:.2f} Gyr")

    print(f"\nAge estimate from BETA (intercept):")
    print(f"  Median age:  {age_results['median_age_beta_gyr']:.2f} Gyr")
    print(f"  16th percentile: {age_results['age_16_beta_gyr']:.2f} Gyr")
    print(f"  84th percentile: {age_results['age_84_beta_gyr']:.2f} Gyr")
    print(f"  1-sigma range: {age_results['median_age_beta_gyr']:.2f} "
          f"+{age_results['age_84_beta_gyr']-age_results['median_age_beta_gyr']:.2f} "
          f"-{age_results['median_age_beta_gyr']-age_results['age_16_beta_gyr']:.2f} Gyr")
    print("="*60 + "\n")


def fnPlotAgeHistograms(age_results):
    """Create age distribution histograms for both alpha and beta"""
    # Alpha histogram
    plt.figure(figsize=(10, 6))
    plt.hist(age_results['ages_from_alpha_gyr'], bins=50, alpha=0.7, color='steelblue',
             edgecolor='black', density=True)
    plt.axvline(age_results['median_age_alpha_gyr'], color='red', linestyle='--',
                linewidth=2, label=f"Median: {age_results['median_age_alpha_gyr']:.2f} Gyr")
    plt.axvline(age_results['age_16_alpha_gyr'], color='orange', linestyle=':',
                linewidth=2, label=f"16th %ile: {age_results['age_16_alpha_gyr']:.2f} Gyr")
    plt.axvline(age_results['age_84_alpha_gyr'], color='orange', linestyle=':',
                linewidth=2, label=f"84th %ile: {age_results['age_84_alpha_gyr']:.2f} Gyr")
    plt.xlabel('Age (Gyr)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('GJ 1132 Age Distribution from Alpha (Slope)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('GJ1132_age_distribution_alpha.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')

    # Beta histogram
    plt.figure(figsize=(10, 6))
    plt.hist(age_results['ages_from_beta_gyr'], bins=50, alpha=0.7, color='coral',
             edgecolor='black', density=True)
    plt.axvline(age_results['median_age_beta_gyr'], color='darkred', linestyle='--',
                linewidth=2, label=f"Median: {age_results['median_age_beta_gyr']:.2f} Gyr")
    plt.axvline(age_results['age_16_beta_gyr'], color='darkorange', linestyle=':',
                linewidth=2, label=f"16th %ile: {age_results['age_16_beta_gyr']:.2f} Gyr")
    plt.axvline(age_results['age_84_beta_gyr'], color='darkorange', linestyle=':',
                linewidth=2, label=f"84th %ile: {age_results['age_84_beta_gyr']:.2f} Gyr")
    plt.xlabel('Age (Gyr)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('GJ 1132 Age Distribution from Beta (Intercept)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('GJ1132_age_distribution_beta.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0.25, facecolor='w')


def fnSaveAgeData(age_results, stellar_mass):
    """Save age estimates to text file"""
    output_filename = 'GJ1132_plausible_ages.txt'
    with open(output_filename, 'w') as f:
        # Write header with metadata and column descriptions
        f.write("# Age estimates for GJ 1132 from flare frequency distribution analysis\n")
        f.write("# Derived using Barnes et al. 2026 fdFlareEquation model\n")
        f.write(f"# Assumed stellar mass: {stellar_mass} M_sun\n")
        f.write(f"# Number of Monte Carlo samples: 10000\n#\n")

        f.write("# Summary statistics from ALPHA (slope):\n")
        f.write(f"# Valid samples: {age_results['n_valid_samples_alpha']}\n")
        f.write(f"# Median age: {age_results['median_age_alpha_gyr']:.4f} Gyr\n")
        f.write(f"# 16th percentile: {age_results['age_16_alpha_gyr']:.4f} Gyr\n")
        f.write(f"# 84th percentile: {age_results['age_84_alpha_gyr']:.4f} Gyr\n")
        f.write(f"# 1-sigma upper: +{age_results['age_84_alpha_gyr']-age_results['median_age_alpha_gyr']:.4f} Gyr\n")
        f.write(f"# 1-sigma lower: -{age_results['median_age_alpha_gyr']-age_results['age_16_alpha_gyr']:.4f} Gyr\n#\n")

        f.write("# Summary statistics from BETA (intercept):\n")
        f.write(f"# Valid samples: {age_results['n_valid_samples_beta']}\n")
        f.write(f"# Median age: {age_results['median_age_beta_gyr']:.4f} Gyr\n")
        f.write(f"# 16th percentile: {age_results['age_16_beta_gyr']:.4f} Gyr\n")
        f.write(f"# 84th percentile: {age_results['age_84_beta_gyr']:.4f} Gyr\n")
        f.write(f"# 1-sigma upper: +{age_results['age_84_beta_gyr']-age_results['median_age_beta_gyr']:.4f} Gyr\n")
        f.write(f"# 1-sigma lower: -{age_results['median_age_beta_gyr']-age_results['age_16_beta_gyr']:.4f} Gyr\n#\n")

        f.write("# Column 1: Age from Alpha (Gyr)\n# Column 2: Age from Alpha (Myr)\n")
        f.write("# Column 3: log(Age from Alpha in Myr)\n# Column 4: Age from Beta (Gyr)\n")
        f.write("# Column 5: Age from Beta (Myr)\n# Column 6: log(Age from Beta in Myr)\n#\n")

        # Write data rows
        max_len = max(len(age_results['ages_from_alpha_gyr']),
                      len(age_results['ages_from_beta_gyr']))
        for i in range(max_len):
            if i < len(age_results['ages_from_alpha_gyr']):
                f.write(f"{age_results['ages_from_alpha_gyr'][i]:.6f}  "
                        f"{age_results['ages_from_alpha_myr'][i]:.6f}  "
                        f"{age_results['log_ages_from_alpha'][i]:.6f}  ")
            else:
                f.write("NaN  NaN  NaN  ")

            if i < len(age_results['ages_from_beta_gyr']):
                f.write(f"{age_results['ages_from_beta_gyr'][i]:.6f}  "
                        f"{age_results['ages_from_beta_myr'][i]:.6f}  "
                        f"{age_results['log_ages_from_beta'][i]:.6f}\n")
            else:
                f.write("NaN  NaN  NaN\n")

    print(f"Saved plausible ages to: {output_filename}")
    print(f"  Total samples from alpha: {len(age_results['ages_from_alpha_gyr'])}")
    print(f"  Total samples from beta:  {len(age_results['ages_from_beta_gyr'])}")


def main():
    """Main analysis workflow - orchestrates GJ 1132 flare analysis"""
    # Download and process TESS data
    lc = flistDownloadTessData()
    totexp = fdCalculateTotalExposure(lc)
    print(f"Total exposure time: {totexp:.2f} days")
    print(f"Number of sectors: {len(lc)}")

    # Get flare parameters and stellar properties
    sectors, t_start, t_stop, lumin, daPeakSigma = ftGetFlareParameters()
    print(f"GJ 1132 log luminosity: {np.log10(lumin.value):.2f}")

    # Compute flare equivalent durations and plot lightcurves
    ed = fdaComputeFlareEquivalentDurations(lc, sectors, t_start, t_stop)
    fnPlotFlareLightcurves(lc, sectors, t_start, t_stop,
                           daPeakSigma=daPeakSigma)

    # Compute and fit FFD
    ffd_x, ffd_y, ffd_xerr, ffd_yerr, fit_results = ftComputeAndFitFfd(
        ed, totexp, lumin, lc, t_start, t_stop)
    fnPrintFfdResults(fit_results)
    alpha, beta = fit_results['alpha'], fit_results['beta']

    # Create all FFD plots
    fnPlotBasicFfd(ffd_x, ffd_y, ffd_yerr, alpha, beta)
    lit_data = fdictGetLiteratureData()
    fnPlotFfdComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta, lit_data)
    cluster_data = fdictGetClusterData()
    fnPlotIlinReproduction(cluster_data)
    fnPlotComprehensiveComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta,
                                 lit_data, cluster_data,
                                 fit_results=fit_results)
    fnPlotAlphaBetaComparison(alpha, beta, fit_results)
    fnPlotProximaComparison(ffd_x, ffd_y, ffd_yerr, alpha, beta, lit_data)
    fnPlotAgeActivityRelation(alpha, beta, lit_data)

    # Run age analysis
    stellar_mass = 0.2
    age_results = fdictRunAgeAnalysis(alpha, beta, fit_results, cluster_data['params'], stellar_mass)
    fnPrintAgeResults(age_results)
    fnPlotAgeHistograms(age_results)
    fnSaveAgeData(age_results, stellar_mass)

    print("\nAnalysis complete! Generated plots:")
    print("  - GJ1132_flares.pdf (3-panel flare plot)")
    print("  - GJ1132_FFD.pdf")
    print("  - GJ1132_FFD_comp.pdf")
    print("  - GJ1132_FFD_comp2.pdf")
    print("  - GJ1132_alpha_beta_comparison.pdf")
    print("  - GJ1132_age_distribution_alpha.pdf")
    print("  - GJ1132_age_distribution_beta.pdf")


def ftRunPipeline(dictKeplerPosterior=None, sFlareJsonPath=None):
    """Run the shared TESS flare analysis pipeline.

    Returns a tuple of all intermediate results needed by any plot function.
    """
    lc = flistDownloadTessData()
    totexp = fdCalculateTotalExposure(lc)
    print(f"Total exposure time: {totexp:.2f} days")
    print(f"Number of sectors: {len(lc)}")

    sectors, t_start, t_stop, lumin, daPeakSigma = ftGetFlareParameters(
        sFlareJsonPath)
    ed = fdaComputeFlareEquivalentDurations(lc, sectors, t_start, t_stop)

    ffd_x, ffd_y, ffd_xerr, ffd_yerr, fit_results = ftComputeAndFitFfd(
        ed, totexp, lumin, lc, t_start, t_stop)
    fnPrintFfdResults(fit_results)

    dAlpha = fit_results['alpha']
    dBeta = fit_results['beta']
    dictLiterature = fdictGetLiteratureData()
    dictCluster = fdictGetClusterData(dictKeplerPosterior=dictKeplerPosterior)

    return (lc, sectors, t_start, t_stop, daPeakSigma, ffd_x, ffd_y,
            ffd_xerr, ffd_yerr, fit_results, dAlpha, dBeta,
            dictLiterature, dictCluster)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="GJ 1132 TESS flare analysis and plotting.")
    parser.add_argument('--plot-flares', metavar='PATH',
                        help="Generate 3-panel flare lightcurve plot.")
    parser.add_argument('--plot-comprehensive', metavar='PATH',
                        help="Generate comprehensive FFD comparison plot.")
    parser.add_argument('--plot-fit-comparison', metavar='PATH',
                        help="Generate Kepler vs TESS alpha-beta comparison.")
    parser.add_argument('--flare-candidates', metavar='PATH',
                        help="Path to flare_candidates.json from "
                             "identifyFlareCandidates.py.")
    parser.add_argument('--kepler-posterior', metavar='PATH',
                        help="Path to kepler_ffd_posterior_stats.json.")
    parser.add_argument('--tess-cache-dir', metavar='PATH',
                        help="Set lightkurve cache directory.")
    args = parser.parse_args()

    bHasPlotFlag = (args.plot_flares or args.plot_comprehensive
                    or args.plot_fit_comparison)

    if args.tess_cache_dir:
        import os
        os.environ['LIGHTKURVE_CACHE_DIR'] = args.tess_cache_dir

    dictKeplerPosterior = None
    if args.kepler_posterior:
        dictKeplerPosterior = fdictLoadKeplerPosterior(args.kepler_posterior)
        print(f"Loaded Kepler posterior from {args.kepler_posterior} "
              f"({dictKeplerPosterior['iNumSamples']} samples)")

    if bHasPlotFlag:
        tPipeline = ftRunPipeline(
            dictKeplerPosterior=dictKeplerPosterior,
            sFlareJsonPath=args.flare_candidates)
        (lc, sectors, t_start, t_stop, daPeakSigma, ffd_x, ffd_y,
         ffd_xerr, ffd_yerr, fit_results, dAlpha, dBeta,
         dictLit, dictCluster) = tPipeline

        if args.plot_flares:
            fnPlotFlareLightcurves(lc, sectors, t_start, t_stop,
                                  daPeakSigma=daPeakSigma,
                                  sOutputPath=args.plot_flares)
        if args.plot_comprehensive:
            fnPlotComprehensiveComparison(ffd_x, ffd_y, ffd_yerr, dAlpha,
                                         dBeta, dictLit, dictCluster,
                                         fit_results=fit_results,
                                         sOutputPath=args.plot_comprehensive)
        if args.plot_fit_comparison:
            fnPlotAlphaBetaComparison(dAlpha, dBeta, fit_results,
                                      sOutputPath=args.plot_fit_comparison,
                                      dictKeplerPosterior=dictKeplerPosterior)
    else:
        main()
