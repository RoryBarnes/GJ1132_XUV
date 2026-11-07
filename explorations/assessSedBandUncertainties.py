#!/usr/bin/env python3
"""Measure how much of each MUSCLES band integral has a catalogued SED uncertainty."""

import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits

S_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S_STEP_DIRECTORY = os.path.join(S_REPOSITORY, "MUSCLESBandConversion")
sys.path.insert(0, S_STEP_DIRECTORY)
import dataDeriveMusclesConversion as conversion  # noqa: E402

DICT_BANDS = {
    "xray": (conversion.fBandXrayRosatLow, conversion.fBandXrayRosatHigh),
    "xuv": (conversion.fBandXuvLow, conversion.fBandXuvHigh),
}


def fdictAssessBand(dictColumns, tBand):
    """Return the catalogued-error coverage and independent-bin error of one band."""
    daOverlap = np.clip(np.minimum(dictColumns["daHigh"], tBand[1])
                        - np.maximum(dictColumns["daLow"], tBand[0]), 0.0, None)
    daFlux = np.nan_to_num(daOverlap * dictColumns["daFlux"])
    daError = np.nan_to_num(daOverlap * dictColumns["daError"])
    dBandFlux = float(np.sum(daFlux))
    dFractionError = float(np.sqrt(np.sum(daError**2))) / dBandFlux
    return {
        "dFractionOfBandWithError": float(np.sum(daFlux[daError > 0])) / dBandFlux,
        "dIndependentBinErrorDex": dFractionError / np.log(10.0),
    }


def fdictReadColumns(sPathFits):
    """Read the wavelength edges and bolometric-normalized flux columns."""
    with fits.open(sPathFits) as hduList:
        data = hduList[1].data
        return {"daLow": np.asarray(data["WAVELENGTH0"], dtype=float),
                "daHigh": np.asarray(data["WAVELENGTH1"], dtype=float),
                "daFlux": np.asarray(data["BOLOFLUX"], dtype=float),
                "daError": np.asarray(data["BOLOERR"], dtype=float)}


def fdictAssessStar(dictTarget):
    """Assess both fitted bands for one star's cached SED."""
    sPath = os.path.join(conversion.sDirectorySeds,
                         conversion.fsBuildSedFileName(dictTarget))
    dictColumns = fdictReadColumns(sPath)
    return {sBand: fdictAssessBand(dictColumns, tBand)
            for sBand, tBand in DICT_BANDS.items()}


def main():
    """Write the per-star assessment to a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sOutputPath", help="Path of the JSON file to write.")
    sOutputPath = parser.parse_args().sOutputPath
    dictResult = {dictTarget["sStar"]: fdictAssessStar(dictTarget)
                  for dictTarget in conversion.listTargets}
    with open(sOutputPath, "w") as fileHandle:
        json.dump(dictResult, fileHandle, indent=2)
    print(f"Saved {sOutputPath}")


if __name__ == "__main__":
    main()
