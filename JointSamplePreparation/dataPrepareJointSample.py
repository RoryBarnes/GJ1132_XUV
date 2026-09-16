#!/usr/bin/env python3
"""
Merge the verbatim Engle paper tables into fit-ready joint samples.

Combines the rotation benchmarks (Engle & Guinan 2023, ApJL 954, L50) with the
X-ray activity samples (Engle 2024, ApJ 960, 62) into one record per star or
cluster ensemble, per M dwarf sub-class. Field stars whose tabulated ages are
rotation-derived carry NO age constraint here: in the joint hierarchical model
their age is latent, informed by the rotation period through the rotation
relation itself. Stars appearing in both papers are merged into a single
record so no constraint is double-counted. Where the two papers adopt
different values for the same star, the rotation (calibration) paper wins and
the alternate is recorded in the notes.

Outputs jointSampleMidLate.json and jointSampleEarly.json.
"""

import csv
import json
import os
import re

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
S_TABLES_DIRECTORY = os.path.join(S_DIRECTORY, "paperTables")

DICT_ROTATION_TO_XRAY_NAME = {
    "Barnard's Star (GJ 699)": "GJ 699",
    "Kapteyn's Star (GJ 191)": "GJ 191",
    "Proxima Cen": "GJ 551",
}

SA_AMBIGUOUS_MIDLATE = [
    "1RXS J031028.8+285952", "LP 195-34", "G 176-13", "HAT 170-05945",
    "HAT 216-04245", "LP 331-57 A", "UCAC4 617-130729", "UCAC3 242-88245",
    "UCAC4 692-060564", "CD-40 5404B (GJ 358)", "GJ 436", "GJ 667 C",
]
SA_AMBIGUOUS_EARLY = [
    "EPIC 201909533", "EPIC 210651981", "GSC 03517-00212",
    "UCAC4 630-046962", "G 80-21", "GSC 02083-01558", "LP 149-56",
    "UCAC4 666-047939", "2MASS J21154192+1746242", "EPIC 210741091",
    "EPIC 202059231", "GJ 832", "BD-18 359", "GJ 433",
]


def fsCanonicalName(sName):
    """Collapse repeated whitespace so names match across the two papers."""
    return re.sub(r"\s+", " ", sName).strip()


def fdParseFloat(sValue):
    """Return float(sValue), or None when the source field is empty."""
    sValue = sValue.strip()
    return float(sValue) if sValue else None


def flistReadTable(sFilename):
    """Read one paperTables CSV into a list of row dictionaries."""
    with open(os.path.join(S_TABLES_DIRECTORY, sFilename)) as fileHandle:
        return list(csv.DictReader(fileHandle))


def fdictAgeConstraintFromRotation(dictRow):
    """Build the independent age constraint carried by a benchmark row."""
    return {
        "sSpace": "linearGyr",
        "dValue": float(dictRow["age_gyr"]),
        "dPlus": float(dictRow["age_plus"]),
        "dMinus": float(dictRow["age_minus"]),
        "sMethod": dictRow["age_via"].split("(")[0].strip(),
    }


def fdictProtFromRotation(dictRow):
    """Build the rotation-period block from a benchmark row."""
    return {
        "dProtDays": float(dictRow["prot_days"]),
        "dPlus": fdParseFloat(dictRow["prot_plus"]),
        "dMinus": fdParseFloat(dictRow["prot_minus"]),
    }


def fdictRotationRecord(dictRow):
    """Build a fit-ready record from one rotation-benchmark CSV row."""
    sVia = dictRow["age_via"]
    return {
        "sName": fsCanonicalName(dictRow["name"]),
        "sSptype": dictRow["sptype"],
        "bSubdwarf": "sd" in dictRow["sptype"],
        "bAmbiguousSubclass": False,
        "bExcludedFromPaperFits": "EXCLUDED" in sVia,
        "bClusterEnsemble": False,
        "sAgeProvenance": "independent",
        "dictAgeConstraint": fdictAgeConstraintFromRotation(dictRow),
        "dictRotation": fdictProtFromRotation(dictRow),
        "dictXray": None,
        "saNotes": [sVia] if "(" in sVia else [],
    }


def fdictXrayBlock(dictRow):
    """Build the X-ray measurement block from an activity-sample row."""
    return {
        "dLogLxLbol": float(dictRow["log_lx_lbol"]),
        "dError": float(dictRow["log_lx_lbol_err"]),
        "dLogLx": fdParseFloat(dictRow["log_lx"]),
        "dLogLxError": fdParseFloat(dictRow["log_lx_err"]),
        "sXraySource": dictRow["xray_src"],
    }


def fdictTabulatedAge(dictRow):
    """Record the activity paper's tabulated age for reference only."""
    return {
        "dLogAgeGyr": float(dictRow["log_age_gyr"]),
        "dLogError": float(dictRow["log_age_err"]),
    }


def fdictXrayOnlyRecord(dictRow, saAmbiguous):
    """Build a record for an activity-sample star absent from the benchmarks."""
    sName = fsCanonicalName(dictRow["star_name"])
    dProt = fdParseFloat(dictRow["prot_days"])
    bEnsemble = dProt is None
    dictAge = fdictClusterEnsembleAge(dictRow) if bEnsemble else None
    dErr = fdParseFloat(dictRow["prot_err"])
    return {
        "sName": sName,
        "sSptype": "",
        "bSubdwarf": False,
        "bAmbiguousSubclass": sName in saAmbiguous,
        "bExcludedFromPaperFits": False,
        "bClusterEnsemble": bEnsemble,
        "sAgeProvenance": "independent" if bEnsemble else "latent",
        "dictAgeConstraint": dictAge,
        "dictRotation": None if bEnsemble else
            {"dProtDays": dProt, "dPlus": dErr, "dMinus": dErr},
        "dictXray": fdictXrayBlock(dictRow),
        "dictTabulatedAge": fdictTabulatedAge(dictRow),
        "saNotes": [],
    }


def fdictClusterEnsembleAge(dictRow):
    """Build the log-space age constraint for an X-ray cluster ensemble row."""
    return {
        "sSpace": "log10Gyr",
        "dValue": float(dictRow["log_age_gyr"]),
        "dError": float(dictRow["log_age_err"]),
        "sMethod": "cluster ensemble (Nunez & Agueros 2016)",
    }


def fnAttachXrayToBenchmark(dictRecord, dictRow):
    """Merge an activity-sample row onto its rotation-benchmark record."""
    dictRecord["dictXray"] = fdictXrayBlock(dictRow)
    dictRecord["dictTabulatedAge"] = fdictTabulatedAge(dictRow)
    dProtXray = fdParseFloat(dictRow["prot_days"])
    dProtBenchmark = dictRecord["dictRotation"]["dProtDays"]
    if dProtXray is not None and abs(dProtXray - dProtBenchmark) > 1e-9:
        dictRecord["saNotes"].append(
            f"activity paper adopts Prot={dProtXray} d (benchmark value "
            f"{dProtBenchmark} d retained)")


def flistBuildJointSample(sRotationFile, sXrayFile, saAmbiguous):
    """Return merged records for one sub-class, benchmarks plus X-ray stars."""
    listRecords = [fdictRotationRecord(r) for r in flistReadTable(sRotationFile)]
    dictByName = {DICT_ROTATION_TO_XRAY_NAME.get(d["sName"], d["sName"]): d
                  for d in listRecords}
    setSeen = set()
    for dictRow in flistReadTable(sXrayFile):
        sName = fsCanonicalName(dictRow["star_name"])
        bEnsemble = fdParseFloat(dictRow["prot_days"]) is None
        if sName in dictByName and not bEnsemble:
            fnAttachXrayToBenchmark(dictByName[sName], dictRow)
            continue
        dictNew = fdictXrayOnlyRecord(dictRow, saAmbiguous)
        if sName in setSeen:
            dictNew["saNotes"].append(
                "duplicate row in the activity paper's table (distinct Prot)")
        setSeen.add(sName)
        listRecords.append(dictNew)
    return listRecords


def fnFlagCrossSubclassStars(listMidLate, listEarly):
    """Note stars the activity paper lists in both sub-class tables."""
    setShared = ({d["sName"] for d in listMidLate}
                 & {d["sName"] for d in listEarly})
    for listRecords in (listMidLate, listEarly):
        for dictRecord in listRecords:
            if dictRecord["sName"] in setShared:
                dictRecord["saNotes"].append(
                    "appears in BOTH sub-class tables of the activity paper")


def fnWriteSample(listRecords, sFilename):
    """Write one joint sample to JSON and print a composition summary."""
    with open(os.path.join(S_DIRECTORY, sFilename), "w") as fileHandle:
        json.dump(listRecords, fileHandle, indent=2)
    iIndependent = sum(d["sAgeProvenance"] == "independent" for d in listRecords)
    iLatent = sum(d["sAgeProvenance"] == "latent" for d in listRecords)
    iMerged = sum(d["dictXray"] is not None and d["dictRotation"] is not None
                  and d["sAgeProvenance"] == "independent" for d in listRecords)
    print(f"{sFilename}: {len(listRecords)} records "
          f"({iIndependent} independent-age, {iLatent} latent-age, "
          f"{iMerged} carrying both an age constraint and X-ray data, "
          f"{sum(d['bSubdwarf'] for d in listRecords)} subdwarfs, "
          f"{sum(d['bAmbiguousSubclass'] for d in listRecords)} ambiguous, "
          f"{sum(d['bExcludedFromPaperFits'] for d in listRecords)} excluded)")


def main():
    """Build and save the fit-ready joint samples for both sub-classes."""
    listMidLate = flistBuildJointSample(
        "rotationMidM.csv", "midlateMXray.csv", SA_AMBIGUOUS_MIDLATE)
    listEarly = flistBuildJointSample(
        "rotationEarlyM.csv", "earlyMXray.csv", SA_AMBIGUOUS_EARLY)
    fnFlagCrossSubclassStars(listMidLate, listEarly)
    fnWriteSample(listMidLate, "jointSampleMidLate.json")
    fnWriteSample(listEarly, "jointSampleEarly.json")


if __name__ == "__main__":
    main()
