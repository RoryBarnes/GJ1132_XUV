"""Integrity checks for the GJ 1132 probabilistic graphical model figure.

Confirms that the PGM structure JSON is internally consistent and that the
rendered figure exists as a valid, non-trivial PDF document. The renderer
writes figures to the workflow plot directory (repo-root ``Plot``); when run
in isolation they land in the step-local ``Plot``. Both locations are checked.
"""

import json
import os

S_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S_REPO_ROOT = os.path.dirname(S_DIRECTORY)
SET_CATEGORIES = {"observed", "stochastic", "deterministic", "constant"}


def fdictLoad(sName):
    """Load a JSON structure file from the step directory."""
    with open(os.path.join(S_DIRECTORY, sName)) as fileHandle:
        return json.load(fileHandle)


def fsFindFigure(sName):
    """Return the first existing path for a figure across candidate dirs."""
    for sBase in (os.path.join(S_REPO_ROOT, "Plot"),
                  os.path.join(S_DIRECTORY, "Plot")):
        sPath = os.path.join(sBase, sName)
        if os.path.exists(sPath):
            return sPath
    return None


def fnAssertValidPdf(sName):
    """Assert a figure exists, is a PDF, and is not a stub."""
    sPath = fsFindFigure(sName)
    assert sPath is not None, f"missing figure {sName}"
    with open(sPath, "rb") as fileHandle:
        baHead = fileHandle.read(5)
    assert baHead == b"%PDF-", f"{sName} is not a valid PDF"
    assert os.path.getsize(sPath) > 3000, f"{sName} is suspiciously small"


def test_pgm_pdf_valid():
    """The probabilistic graphical model renders to a valid PDF."""
    fnAssertValidPdf("ProbabilisticGraphicalModel.pdf")


def test_pgm_structure_consistent():
    """PGM edges reference declared nodes and use valid categories."""
    dictPgm = fdictLoad("probabilisticGraphicalModelStructure.json")
    setIds = {dictNode["sId"] for dictNode in dictPgm["listNodes"]}
    for dictNode in dictPgm["listNodes"]:
        assert dictNode["sCategory"] in SET_CATEGORIES
    for dictEdge in dictPgm["listEdges"]:
        assert dictEdge["sSrc"] in setIds and dictEdge["sDst"] in setIds


def test_pgm_has_every_category():
    """The PGM exercises all four RevBayes node categories."""
    dictPgm = fdictLoad("probabilisticGraphicalModelStructure.json")
    setPresent = {dictNode["sCategory"] for dictNode in dictPgm["listNodes"]}
    assert setPresent == SET_CATEGORIES


def test_pgm_categorization_table_present():
    """A parameter-categorization record accompanies the PGM structure."""
    dictPgm = fdictLoad("probabilisticGraphicalModelStructure.json")
    assert len(dictPgm["listCategorization"]) >= 4
    for dictRow in dictPgm["listCategorization"]:
        assert dictRow["sCategory"] in SET_CATEGORIES
        assert dictRow["sRationale"]
