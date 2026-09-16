"""Integrity checks for the merged Engle joint samples."""

import json
import os
import sys

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S_UTILITIES_DIRECTORY = os.path.join(os.path.dirname(S_DIRECTORY), "utils")
sys.path.insert(0, S_UTILITIES_DIRECTORY)


def fdictLoadJson(sName):
    """Load a JSON product from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


def test_joint_samples_composition():
    """The merged samples carry the documented counts and provenance flags."""
    listMidLate = fdictLoadJson("jointSampleMidLate.json")
    assert len(listMidLate) == 119
    assert sum(d["bSubdwarf"] for d in listMidLate) == 5
    assert sum(d["bExcludedFromPaperFits"] for d in listMidLate) == 1
    iMerged = sum(d["dictXray"] is not None and d["dictRotation"] is not None
                  and d["sAgeProvenance"] == "independent" for d in listMidLate)
    assert iMerged == 5


def test_fit_filter_reproduces_paper_composition():
    """The composition filter yields the documented benchmark counts."""
    import dataRefitJointRelations as m
    sMidLate = os.path.join(S_DIRECTORY, "jointSampleMidLate.json")
    sEarly = os.path.join(S_DIRECTORY, "jointSampleEarly.json")
    assert m.fdictBuildData(sMidLate, "midlate", False)["iBenchmarks"] == 29
    assert m.fdictBuildData(sMidLate, "midlate", True)["iBenchmarks"] == 34
    assert m.fdictBuildData(sEarly, "early", False)["iBenchmarks"] == 21
    assert m.fdictBuildData(sEarly, "early", True)["iBenchmarks"] == 24
