"""Shared utilities for cumulative XUV flux analysis."""

import json

import numpy as np
import pandas as pd
import vplot


D_CUMULATIVE_EARTH_FLUX = 9.759583e+15
D_SHORELINE_FLUX = 51.43
D_LOWER_BOUND = 20
D_UPPER_BOUND = 4e3
D_CONFIDENCE_INTERVAL = 95
I_NUM_BINS = 50


def fdictLoadConvergedJson(sFilePath):
    """Load a vconverge Converged_Param_Dictionary.json file."""
    with open(sFilePath, 'r') as fileHandle:
        sContent = fileHandle.read().strip()
        if sContent.startswith('"') and sContent.endswith('"'):
            sContent = sContent[1:-1]
            sContent = sContent.replace('\\"', '"')
    dictData = json.loads(sContent)
    if not isinstance(dictData, dict):
        raise ValueError(f"Loaded data from {sFilePath} is not a dictionary.")
    return dictData


def daExtractFluxValues(dictData, sKey="b,CumulativeXUVFlux,final"):
    """Extract, normalize, and filter cumulative XUV flux values."""
    daRawFlux = dictData.get(sKey)
    if daRawFlux is None:
        raise ValueError(f"Key '{sKey}' not found in data.")
    daCumulativeXUVFlux = np.array(pd.Series(daRawFlux).dropna())
    daCumulativeXUVFlux = daCumulativeXUVFlux / D_CUMULATIVE_EARTH_FLUX
    baMask = ((daCumulativeXUVFlux >= D_LOWER_BOUND)
              & (daCumulativeXUVFlux <= D_UPPER_BOUND))
    daCumulativeXUVFlux = daCumulativeXUVFlux[baMask]
    if len(daCumulativeXUVFlux) == 0:
        raise ValueError("No flux values remain after filtering.")
    return daCumulativeXUVFlux


def ftComputeStatistics(daSamples):
    """Return (dMean, dLower, dUpper) for the confidence interval."""
    dMean = np.mean(daSamples)
    dMin = (100 - D_CONFIDENCE_INTERVAL) / 2
    dMax = D_CONFIDENCE_INTERVAL + dMin
    dLower = np.percentile(daSamples, dMin)
    dUpper = np.percentile(daSamples, dMax)
    return dMean, dLower, dUpper


def ftComputeLogBins(daSamples):
    """Return (daBinCenters, daFractions) for a log-spaced histogram."""
    dLogLower = np.log10(D_LOWER_BOUND)
    dLogUpper = np.log10(D_UPPER_BOUND)
    dBinWidth = (dLogUpper - dLogLower) / I_NUM_BINS
    daLogBinEdges = np.arange(dLogLower, dLogUpper + dBinWidth, dBinWidth)
    daBinEdges = 10**daLogBinEdges
    daCounts, _ = np.histogram(daSamples, bins=daBinEdges)
    daFractions = daCounts / len(daSamples)
    daBinCenters = np.sqrt(daBinEdges[:-1] * daBinEdges[1:])
    return daBinCenters, daFractions


def ftGatherFluxes(sDirectory):
    """Load, bin, and summarize cumulative XUV flux data.

    Returns (daBinCenters, daFractions, dMean, dLower, dUpper).
    """
    sFilePath = sDirectory + '/output/Converged_Param_Dictionary.json'
    dictData = fdictLoadConvergedJson(sFilePath)
    daFlux = daExtractFluxValues(dictData)
    dMean, dLower, dUpper = ftComputeStatistics(daFlux)
    daBinCenters, daFractions = ftComputeLogBins(daFlux)
    return daBinCenters, daFractions, dMean, dLower, dUpper
