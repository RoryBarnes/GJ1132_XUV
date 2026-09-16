"""Integrity checks for the catalog-ready coefficient package."""

import json
import os

import numpy as np
import pytest

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SA_TAGS = ["midlate_withsd", "midlate_nosd", "early_withsd", "early_nosd"]


def fdictLoadPackage():
    """Load the packaged coefficient posteriors from the step directory."""
    sPath = os.path.join(S_DIRECTORY, "engleCoefficientPosteriors.json")
    with open(sPath) as fileHandle:
        return json.load(fileHandle)


def test_package_contains_all_variants_and_conversion():
    """All four fit variants and the conversion block are present."""
    dictPackage = fdictLoadPackage()
    for sTag in SA_TAGS:
        assert sTag in dictPackage, f"missing fit variant {sTag}"
    dictConversion = dictPackage["dictConversionXuv"]
    assert dictConversion["dSlope"] > 0
    assert np.array(dictConversion["daCovariance"]).shape == (2, 2)


@pytest.mark.parametrize("sTag", SA_TAGS)
def test_variant_posterior_block_is_valid(sTag):
    """Each variant carries a 12-vector mean, a valid 12x12 covariance,
    a converged chain, and the correct canonical flag."""
    dictVariant = fdictLoadPackage()[sTag]
    assert dictVariant["bCanonical"] == sTag.endswith("_withsd")
    assert len(dictVariant["daPosteriorMean"]) == 12
    daCovariance = np.array(dictVariant["daPosteriorCovariance"])
    assert daCovariance.shape == (12, 12)
    assert np.allclose(daCovariance, daCovariance.T)
    assert np.all(np.linalg.eigvalsh(daCovariance) > 0)
    assert dictVariant["dictConvergence"]["bConverged"]
