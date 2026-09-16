# GJ 1132 XUV Evolution Pipeline

This repository models the XUV radiation environment of the exoplanet GJ 1132 b to constrain its atmospheric evolution and habitability.

## Scientific Context

GJ 1132 b is a rocky exoplanet orbiting an M dwarf. The cumulative XUV flux it receives over billions of years determines whether it could retain liquid water. This pipeline infers posterior distributions for model parameters using a combination of Kepler flare statistics, TESS observations, and Bayesian inference.

## Pipeline Overview

The workflow has 13 steps organized in a dependency chain:

1. **Kepler FFD Corner** — MCMC inference of flare frequency distribution parameters from Kepler ensemble data (Davenport et al. 2019)
2. **FFD Age Comparison** — Plot FFD evolution across stellar ages
3. **TESS Flare Candidates** (interactive) — Identify flare candidates in GJ 1132 TESS lightcurves via sigma-clipping with human review
4. **TESS Flare Lightcurves** — Plot identified flares from TESS data
5. **Comprehensive FFD Comparison** — Compare Kepler and TESS flare statistics
6. **XUV Luminosity Distribution** — Monte Carlo sampling of L_XUV using Ribas et al. scaling
7. **Engle Age Distribution** — Monte Carlo sampling of stellar age using Engle & Guinan gyrochronology
8. **Maximum Likelihood** — MaxLEV estimation of vplanet model parameters
9. **Bayesian Posteriors** — alabi surrogate-model Bayesian inference with dynesty, emcee, multinest, ultranest
10. **XUV Evolution** — vconverge convergence testing for XUV evolution trajectories
11. **Cumulative XUV and Cosmic Shoreline** — Final cumulative XUV flux distributions and cosmic shoreline placement
12. **Error Source Comparison** — Decompose error contributions (stellar vs model)
13. **Kepler vs TESS Comparison** — Direct comparison of Kepler and TESS flare fits

## Key Dependencies

- Steps 04, 05, 13 depend on Step 03 (interactive TESS flare identification)
- Steps 10-12 depend on Steps 01, 07, 09 (MCMC samples, age samples, dynesty posteriors)
- Step 09 depends on Step 06 (XUV luminosity constraints)

## Tools Used

- `vplanet` — Planetary evolution forward model (C binary)
- `maxlev` — Maximum likelihood estimator for vplanet
- `vconverge` — Convergence testing for vplanet parameter sweeps
- `alabi` — Machine learning surrogate model for Bayesian inference
- `lightkurve` — TESS/Kepler lightcurve analysis
- `vplot` — Standardized scientific plotting (STIX Two Text font, accessible colors)

## Data Products

- `flare_mcmc_samples.npy` — MCMC posterior samples for FFD parameters
- `kepler_ffd_posterior_stats.json` — Summary statistics of FFD posteriors
- `flare_candidates.json` — Labeled TESS flare candidates
- `lxuv_constraints.json` — XUV luminosity constraints for Bayesian inference
- `dynesty_transform_final.npy` — Dynesty posterior samples in vspace format
- `Converged_Param_Dictionary.json` — vconverge convergence results

## Conventions

- Plot scripts are prefixed with `plot` (e.g., `plotCornerVariableSlope.py`)
- Data analysis scripts are prefixed with `data` (e.g., `dataKeplerFfd.py`)
- All figures are PDF by default, stored in `Plot/` subdirectories
- Reference PNGs for figure comparison are stored at the repo root level
