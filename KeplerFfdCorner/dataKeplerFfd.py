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
from scipy.optimize import minimize, differential_evolution

def fdaLogFlareRateModel(log_energy, log_age, mass, params):
    """
    Model for log10(flare rate) based on Davenport et al. 2019 Equation (3).
    
    From Eq. (3): log10(rate) = (a1*log_age + a2*mass + a3) * log_energy + (b1*log_age + b2*mass + b3)
    
    Where the "a" terms control the slope (energy dependence) and "b" terms control the intercept.
    
    Parameters:
    -----------
    log_energy : float or array
        Base-10 logarithm of flare energy (erg)
    log_age : float or array  
        Base-10 logarithm of stellar age (Gyr)
    mass : float or array
        Stellar mass (solar masses)
    params : array-like
        Model parameters [a1, a2, a3, b1, b2, b3]
        
    Returns:
    --------
    log_rate : float or array
        Base-10 logarithm of predicted flare rate
    """
    a1, a2, a3, b1, b2, b3 = params
    
    # Slope term (depends on age and mass) - controls energy dependence
    slope = a1 * log_age + a2 * mass + a3
    
    # Intercept term (depends on age and mass) - controls normalization
    intercept = b1 * log_age + b2 * mass + b3
    
    # Linear model: log(rate) = slope * log(energy) + intercept
    log_rate = slope * log_energy + intercept
    
    return log_rate

def fdLogLikelihoodEnsemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Log-likelihood function for the ensemble flare frequency model.
    
    This version handles data from multiple stars properly by computing
    the likelihood for the entire ensemble dataset.
    
    Parameters:
    -----------
    params : array-like
        Model parameters [a1, a2, a3, b1, b2, b3]
    log_energy : array
        Base-10 logarithm of flare energies
    log_age : array
        Base-10 logarithm of stellar ages
    mass : array
        Stellar masses
    observed_log_ff : array
        Observed log10(flare frequency)
    log_ff_errors : array
        Uncertainties in log10(flare frequency)
        
    Returns:
    --------
    log_like : float
        Log-likelihood value
    """
    # Predict flare rates for all data points
    predicted_log_ff = fdaLogFlareRateModel(log_energy, log_age, mass, params)
    
    # Calculate residuals
    residuals = observed_log_ff - predicted_log_ff
    
    # Gaussian likelihood
    chi_squared = np.sum((residuals / log_ff_errors) ** 2)
    log_like = -0.5 * (chi_squared + np.sum(np.log(2 * np.pi * log_ff_errors ** 2)))
    
    # Check for numerical issues
    if not np.isfinite(log_like):
        return -np.inf
    
    return log_like

def fdLogPriorEnsemble(params):
    """
    Log-prior function with physically motivated priors for ensemble analysis.
    
    Parameters:
    -----------
    params : array-like
        Model parameters [a1, a2, a3, b1, b2, b3]
        Where a1, a2, a3 control slope (energy dependence)
        And b1, b2, b3 control intercept (normalization)
        
    Returns:
    --------
    log_p : float
        Log-prior probability
    """
    a1, a2, a3, b1, b2, b3 = params
    
    # Priors based on correct parameter roles from Davenport et al. 2019
    if (-5 < a1 < 5 and             # Age dependence of slope
        -5 < a2 < 5 and             # Mass dependence of slope
        -3 < a3 < 0 and             # Baseline slope (negative for power-law)
        -20 < b1 < 20 and           # Age dependence of intercept
        -20 < b2 < 20 and           # Mass dependence of intercept  
        -30 < b3 < 30):             # Baseline intercept (broad range)
        return 0.0  # Uniform prior
    else:
        return -np.inf  # Outside prior bounds

def fdLogPosteriorEnsemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Log-posterior probability function for ensemble analysis.
    """
    lp = fdLogPriorEnsemble(params)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = fdLogLikelihoodEnsemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    
    return lp + ll

def fdaFindInitialGuess(log_energy, log_age, mass, observed_log_ff, log_ff_errors):
    """
    Find initial parameter guess using maximum likelihood estimation for ensemble data.
    
    Returns:
    --------
    initial_params : array
        Best-fit parameters to use as starting point for MCMC
    """
    
    def neg_log_likelihood(params):
        return -fdLogLikelihoodEnsemble(params, log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    
    # Better initial guess based on corrected parameter roles
    # a1, a2, a3 control slope; b1, b2, b3 control intercept
    p0 = [0.0, 0.0, -1.5, -2.0, 5.0, 15.0]  # More physical starting point
    
    # Bounds for optimization (matching the corrected priors)
    bounds = [(-5, 5), (-5, 5), (-3, 0), (-20, 20), (-20, 20), (-30, 30)]
    
    print("Optimizing initial parameters...")
    try:
        # Use differential evolution for more robust global optimization
        result = differential_evolution(neg_log_likelihood, bounds, seed=42, maxiter=500)
        
        if result.success:
            print(f"Initial optimization successful. Final likelihood: {-result.fun:.2f}")
            return result.x
        else:
            print("Warning: Initial optimization did not converge fully.")
            return result.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Using default initial guess")
        return np.array(p0)

def ftLoadEnsembleData(data_file):
    """
    Load and process the ensemble flare frequency data.
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file with ensemble flare data
        
    Returns:
    --------
    data_arrays : tuple
        (log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids)
    """
    
    print("Loading ensemble data...")
    try:
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} rows from {data_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}. Using sample data for demonstration...")
        # Use sample data if file not found
        data_str = """logE,logAge,mass,giclr,Prot,FF,FFerr,logFF,logFFerr
35.34892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.0025314096480615386,0.0021921240538215455,-2.596637568987615,inf
35.148920207016076,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.007036270286523077,0.002536582324484261,-2.1526574861999483,inf
34.94892020701607,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.023907653795607695,0.004147031799937468,-1.6214630417686646,inf
34.74892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.06483541443878463,0.005795819712855208,-1.1881877087272184,inf
34.548920207016074,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.14304210179886157,0.007897060298653792,-0.8445361171657367,inf
34.34892020701608,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.25889511416534616,0.009571461421119146,-0.5868761454363197,inf
34.148920207016076,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.3392218030355154,0.008164820017038414,-0.46951624181124124,inf
33.94892020701607,-0.3258177697883767,0.6905162489312976,1.6259999999999994,0.244,0.35545834820083205,0.003077491859003192,-0.449211281590533,inf
34.5777686221268,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0021430028144,0.003738156698102526,-2.668977258599988,inf
34.377768622126794,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0043695185408999995,0.0032664607840441474,-2.3595664134986,inf
34.1777686221268,-0.3214820754570555,0.803880852947368,1.0490000000000013,0.215,0.0043695185408999995,0.0026240314624560173,-2.3595664134986,inf"""
        
        from io import StringIO
        data = pd.read_csv(StringIO(data_str))
        print(f"Using sample data with {len(data)} rows")
    
    print("Processing ensemble data...")
    
    # Filter out infinite, zero, and invalid flare frequency values
    valid_mask = (
        (data['FF'] > 0) & 
        np.isfinite(data['FF']) & 
        np.isfinite(data['FFerr']) & 
        (data['FFerr'] > 0) &
        np.isfinite(data['logAge']) &
        np.isfinite(data['mass']) &
        np.isfinite(data['logE'])
    )
    
    data_clean = data[valid_mask].copy()
    
    if len(data_clean) == 0:
        raise ValueError("No valid flare frequency data found!")
    
    # Create unique star identifiers based on stellar properties
    # Stars with same age, mass, and rotation period are considered the same
    stellar_props = ['logAge', 'mass', 'Prot']
    data_clean['star_id'] = data_clean.groupby(stellar_props).ngroup()
    
    # Get ensemble statistics
    n_stars = data_clean['star_id'].nunique()
    n_total_points = len(data_clean)
    
    print(f"Ensemble statistics:")
    print(f"  Total valid data points: {n_total_points}")
    print(f"  Number of stars: {n_stars}")
    print(f"  Average points per star: {n_total_points/n_stars:.1f}")
    
    # Extract variables for the full ensemble
    log_energy = data_clean['logE'].values
    log_age = data_clean['logAge'].values  
    mass = data_clean['mass'].values
    observed_ff = data_clean['FF'].values
    ff_errors = data_clean['FFerr'].values
    star_ids = data_clean['star_id'].values
    
    # Convert to log space for fitting
    observed_log_ff = np.log10(observed_ff)
    
    # Estimate errors in log space (using error propagation)
    log_ff_errors = ff_errors / (observed_ff * np.log(10))
    
    # Set minimum error floor to avoid numerical issues
    min_log_error = 0.01  # ~2.3% relative error
    log_ff_errors = np.maximum(log_ff_errors, min_log_error)
    
    print(f"Data ranges:")
    print(f"  Energy: {log_energy.min():.2f} to {log_energy.max():.2f} (log10 erg)")
    print(f"  Age: {log_age.min():.3f} to {log_age.max():.3f} (log10 Gyr)")
    print(f"  Mass: {mass.min():.2f} to {mass.max():.2f} (solar masses)")
    print(f"  Flare rate: {observed_log_ff.min():.2f} to {observed_log_ff.max():.2f} (log10 events/day)")
    
    # Check for potential issues
    age_range = log_age.max() - log_age.min()
    mass_range = mass.max() - mass.min()
    
    if age_range < 0.1:
        print(f"Warning: Limited age range ({age_range:.3f} dex). Age evolution may be poorly constrained.")
    if mass_range < 0.1:
        print(f"Warning: Limited mass range ({mass_range:.2f} M☉). Mass dependence may be poorly constrained.")
    
    return log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids

def fnRunMcmcEnsemble(data_file, nwalkers=32, nsteps=5000, burn_in=1000, thin=10):
    """
    Run MCMC sampling to infer flare frequency parameters from ensemble data.
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file with ensemble flare data
    nwalkers : int
        Number of MCMC walkers
    nsteps : int  
        Number of MCMC steps
    burn_in : int
        Number of burn-in steps to discard
    thin : int
        Thinning factor for chain
        
    Returns:
    --------
    sampler : emcee.EnsembleSampler
        MCMC sampler object with results
    """
    
    # Load and process ensemble data
    log_energy, log_age, mass, observed_log_ff, log_ff_errors, star_ids = ftLoadEnsembleData(data_file)
    
    # Find initial parameter estimate
    print("Finding initial parameter guess...")
    initial_params = fdaFindInitialGuess(log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    print(f"Initial parameters: a1={initial_params[0]:.3f}, a2={initial_params[1]:.3f}, a3={initial_params[2]:.3f}")
    print(f"                   b1={initial_params[3]:.3f}, b2={initial_params[4]:.3f}, b3={initial_params[5]:.3f}")
    
    # Set up MCMC
    ndim = 6  # Number of parameters
    
    # Initialize walkers around the best-fit solution
    # Use larger scatter for ensemble analysis
    pos = initial_params + 0.1 * np.random.randn(nwalkers, ndim)
    
    # Ensure all walkers start within prior bounds
    for i in range(nwalkers):
        while fdLogPriorEnsemble(pos[i]) == -np.inf:
            pos[i] = initial_params + 0.1 * np.random.randn(ndim)
    
    # Create sampler for ensemble analysis
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, fdLogPosteriorEnsemble, 
        args=(log_energy, log_age, mass, observed_log_ff, log_ff_errors)
    )
    
    # Run burn-in
    print(f"Running burn-in with {burn_in} steps...")
    pos, _, _ = sampler.run_mcmc(pos, burn_in, progress=True)
    sampler.reset()
    
    # Run production
    print(f"Running MCMC with {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Check convergence
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation times: {tau}")
        
        if np.any(tau * 50 > nsteps):
            print("Warning: Chain may not be converged. Consider running longer.")
        else:
            print("Chain appears to be well converged.")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")
        print("This is often due to short chains. Results should still be usable.")
    
    return sampler

def fsSaveSamples(samples, param_names, filename='flare_mcmc_samples.txt'):
    """
    Save MCMC samples to file with header information.
    
    Parameters:
    -----------
    samples : array
        MCMC samples array (n_samples, n_parameters)
    param_names : list
        Names of parameters
    filename : str
        Output filename
    """
    
    print(f"Saving {len(samples)} MCMC samples to {filename}...")
    
    # Create header with parameter information
    header = f"MCMC samples from stellar flare frequency analysis\n"
    header += f"Based on Davenport et al. 2019 (ApJ 871, 241) Equation (3)\n"
    header += f"Model: log10(rate) = (a1*log_age + a2*mass + a3) * log_energy + (b1*log_age + b2*mass + b3)\n"
    header += f"Columns: {' '.join(param_names)}\n"
    header += f"Number of samples: {len(samples)}\n"
    header += f"Parameter roles:\n"
    header += f"  a1, a2, a3: Control energy slope (power-law index)\n"
    header += f"  b1, b2, b3: Control normalization (intercept)\n"
    
    # Save with numpy
    np.savetxt(filename, samples, 
               header=header,
               fmt='%.6f',
               delimiter='\t')
    
    print(f"Samples saved successfully!")
    print(f"File contains {samples.shape[1]} parameters and {samples.shape[0]} samples")
    
    return filename

def fdictSaveMultipleFormats(samples, param_names, base_filename='flare_mcmc_samples'):
    """
    Save MCMC samples in multiple formats for different use cases.
    
    Parameters:
    -----------
    samples : array
        MCMC samples array
    param_names : list
        Parameter names
    base_filename : str
        Base filename (extensions will be added)
    
    Returns:
    --------
    filenames : dict
        Dictionary of saved filenames
    """
    
    filenames = {}
    
    # 1. Plain text file (easy to read, good for plotting)
    txt_file = f"{base_filename}.txt"
    fsSaveSamples(samples, param_names, txt_file)
    filenames['txt'] = txt_file
    
    # 2. CSV file (easy to load in Excel, R, etc.)
    csv_file = f"{base_filename}.csv"
    print(f"Saving samples to {csv_file}...")
    
    import pandas as pd
    df = pd.DataFrame(samples, columns=param_names)
    df.to_csv(csv_file, index=False, float_format='%.6f')
    filenames['csv'] = csv_file
    
    # 3. NumPy binary file (most efficient, preserves full precision)
    npy_file = f"{base_filename}.npy"
    print(f"Saving samples to {npy_file}...")
    np.save(npy_file, samples)
    filenames['npy'] = npy_file
    
    # 4. Save parameter names separately
    params_file = f"{base_filename}_param_names.txt"
    with open(params_file, 'w') as f:
        f.write("# Parameter names for MCMC samples\n")
        f.write("# From Davenport et al. 2019 Eq. (3)\n")
        f.write("# a1, a2, a3: energy slope parameters\n")
        f.write("# b1, b2, b3: normalization parameters\n")
        for i, name in enumerate(param_names):
            f.write(f"{i}\t{name}\n")
    filenames['params'] = params_file
    
    print(f"\nSaved samples in {len(filenames)} formats:")
    for fmt, filename in filenames.items():
        print(f"  {fmt.upper()}: {filename}")
    
    return filenames
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
        print("Stellar Flare Frequency Ensemble MCMC Analysis")
        print("=" * 50)
        sampler = fnRunMcmcEnsemble(
            "ensemble_FFD.csv", nwalkers=32, nsteps=8000, burn_in=1000
        )
        daSamples = fdaAnalyzeResults(sampler)

    print("\nData analysis complete!")
    print("Run plotKeplerFfd.py to generate figures.")