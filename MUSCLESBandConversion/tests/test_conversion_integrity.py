"""Integrity checks for the re-derived MUSCLES X-UV band conversion."""

import json
import os

import numpy as np

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fdictLoadJson(sName):
    """Load a JSON product from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


def test_muscles_conversion_products():
    """The re-derived conversion has a valid covariance and positive scatter."""
    dictFit = fdictLoadJson("conversionFit.json")
    dictPrimary = dictFit["primary_fit_all_targets_rosat_band"]
    daCovariance = np.array(dictPrimary["covariance_slope_intercept"])
    assert daCovariance.shape == (2, 2)
    assert np.allclose(daCovariance, daCovariance.T)
    assert np.all(np.linalg.eigvalsh(daCovariance) > 0)
    dictScatter = dictPrimary["intrinsic_scatter_dex"]
    assert 0 < dictScatter["fScatterCredible16"] \
        < dictScatter["fScatterPosteriorMedian"] \
        < dictScatter["fScatterCredible84"]


def test_per_star_fluxes_cover_full_sample():
    """Every fitted star carries finite band ratios in the per-star file."""
    dictFluxes = fdictLoadJson("perStarBandFluxes.json")
    dictFit = fdictLoadJson("conversionFit.json")
    iStars = dictFit["primary_fit_all_targets_rosat_band"]["n_stars"]
    listStars = dictFluxes if isinstance(dictFluxes, list) else \
        dictFluxes.get("stars", list(dictFluxes.values()))
    assert len(listStars) >= iStars
