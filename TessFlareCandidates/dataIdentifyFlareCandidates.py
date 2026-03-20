#!/usr/bin/env python
"""Interactive flare candidate identification for GJ 1132 TESS data.

Scans all TESS sectors for flare-like brightness excursions using a
MAD-based robust sigma threshold (Rousseeuw & Croux 1993, JASA 88,
1273), then presents each candidate for interactive classification.
"""

import argparse
import json
import os
import select
import sys
import tkinter as tk

import lightkurve as lk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image, ImageTk

import vplot

D_DEFAULT_SIGMA_THRESHOLD = 2.5
I_DEFAULT_WINDOW_LENGTH = 101
I_GAP_TOLERANCE = 2
I_MIN_CONSECUTIVE = 2
I_DECAY_BASELINE_COUNT = 3


# ===== Data Loading =====

def flistDownloadTessData():
    """Download TESS lightcurve data for GJ 1132."""
    print("Downloading TESS lightcurve data for GJ 1132...")
    return lk.search_lightcurve(
        'GJ 1132', mission='TESS', author='SPOC', exptime=120
    ).download_all()


# ===== Detection =====

def fdComputeRobustSigma(daFlux):
    """Return MAD-based robust sigma (Rousseeuw & Croux 1993, JASA 88)."""
    daFinite = daFlux[np.isfinite(daFlux)]
    dMedian = np.median(daFinite)
    return 1.4826 * np.median(np.abs(daFinite - dMedian))


def flistGroupConsecutiveIndices(iaIndices):
    """Group indices into clusters, merging gaps <= I_GAP_TOLERANCE."""
    if len(iaIndices) == 0:
        return []
    listGroups = []
    listCurrent = [iaIndices[0]]
    for i in range(1, len(iaIndices)):
        if iaIndices[i] - iaIndices[i - 1] <= I_GAP_TOLERANCE:
            listCurrent.append(iaIndices[i])
        else:
            listGroups.append(np.array(listCurrent))
            listCurrent = [iaIndices[i]]
    listGroups.append(np.array(listCurrent))
    return listGroups


def fiExtendDecayTail(daFlux, iLastAbove, dMedian, dSigma):
    """Return index where the decay tail returns to baseline."""
    iConsecutiveBelow = 0
    for i in range(iLastAbove + 1, len(daFlux)):
        if daFlux[i] < dMedian + dSigma:
            iConsecutiveBelow += 1
            if iConsecutiveBelow >= I_DECAY_BASELINE_COUNT:
                return i - I_DECAY_BASELINE_COUNT + 1
        else:
            iConsecutiveBelow = 0
    return min(iLastAbove + 1, len(daFlux) - 1)


def fdComputeEquivalentDuration(daFlux, daTime, iStart, iStop):
    """Integrate (flux - 1) * dt over the candidate window (seconds)."""
    daSliceFlux = daFlux[iStart:iStop + 1] - 1.0
    daSliceTime = daTime[iStart:iStop + 1] * 86400.0
    daTrapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(daTrapezoid(daSliceFlux, daSliceTime))


def fdictBuildCandidate(daFluxNorm, daFluxFlat, daTime, iaGroup,
                        iExtended, iSectorIndex, iSectorNumber,
                        dMedian, dSigma):
    """Build a candidate dictionary from a detected group."""
    iPeak = iaGroup[np.argmax(daFluxFlat[iaGroup])]
    iStart = iaGroup[0]
    iStop = max(iaGroup[-1], iExtended)
    dDuration = (daTime[iStop] - daTime[iStart]) * 1440.0
    dEquivDur = fdComputeEquivalentDuration(daFluxNorm, daTime, iStart, iStop)
    return {
        'iSectorIndex': iSectorIndex,
        'iSectorNumber': iSectorNumber,
        'dTimeStart': float(daTime[iStart]),
        'dTimeStop': float(daTime[iStop]),
        'dTimePeak': float(daTime[iPeak]),
        'dPeakFlux': float(daFluxNorm[iPeak]),
        'dPeakSigma': round(float((daFluxFlat[iPeak] - dMedian) / dSigma), 2),
        'dDurationMinutes': round(float(dDuration), 1),
        'dEquivalentDuration': round(float(dEquivDur), 2),
        'iNumPointsAbove': len(iaGroup),
        'sLabel': '',
    }


def flistDetectCandidatesInSector(lcNorm, lcFlat, iSectorIndex,
                                  iSectorNumber, dSigmaThreshold):
    """Detect flare candidates in one sector's lightcurve."""
    daFluxFlat = np.array(lcFlat['flux'].value, dtype=float)
    daFluxNorm = np.array(lcNorm['flux'].value, dtype=float)
    daTime = np.array(lcNorm['time'].value, dtype=float)
    baMask = np.isfinite(daFluxFlat) & np.isfinite(daFluxNorm)
    dMedian = np.median(daFluxFlat[baMask])
    dSigma = fdComputeRobustSigma(daFluxFlat[baMask])
    dThreshold = dMedian + dSigmaThreshold * dSigma
    iaAbove = np.where(baMask & (daFluxFlat > dThreshold))[0]
    listGroups = flistGroupConsecutiveIndices(iaAbove)
    listCandidates = []
    for iaGroup in listGroups:
        if len(iaGroup) < I_MIN_CONSECUTIVE:
            continue
        iExtended = fiExtendDecayTail(daFluxFlat, iaGroup[-1], dMedian,
                                      dSigma)
        listCandidates.append(fdictBuildCandidate(
            daFluxNorm, daFluxFlat, daTime, iaGroup, iExtended,
            iSectorIndex, iSectorNumber, dMedian, dSigma))
    return listCandidates


def flistDetectAllCandidates(listLightcurves, dSigmaThreshold):
    """Detect candidates across all sectors, sorted by peak sigma."""
    listAll = []
    for k in range(len(listLightcurves)):
        lcNorm = listLightcurves[k].normalize()
        try:
            lcFlat = lcNorm.flatten(window_length=I_DEFAULT_WINDOW_LENGTH)
        except Exception as e:
            iSector = listLightcurves[k].meta.get('SECTOR', k)
            print(f"  Sector {iSector}: flatten failed ({e}), skipping")
            continue
        iSector = listLightcurves[k].meta.get('SECTOR', k)
        listSector = flistDetectCandidatesInSector(
            lcNorm, lcFlat, k, iSector, dSigmaThreshold)
        print(f"  Sector {iSector}: {len(listSector)} candidates")
        listAll.extend(listSector)
    listAll.sort(key=lambda d: d['dPeakSigma'], reverse=True)
    return listAll


# ===== Display =====

def fnVerifyInteractiveDisplay():
    """Exit with an error if an X11 display is not available."""
    sDisplay = os.environ.get('DISPLAY', '')
    if not sDisplay:
        print("ERROR: Cannot open interactive figure window.")
        print("  DISPLAY environment variable is not set.")
        print("\nThis script requires X11 display forwarding.")
        print("If running in a container, ensure X11 is enabled.")
        sys.exit(1)


def fnRenderFigureToTk(fig, tkRoot):
    """Render a matplotlib figure and display it in a Tk window.

    Returns the ImageTk.PhotoImage (caller must retain reference).
    """
    fig.canvas.draw()
    daBuffer = fig.canvas.buffer_rgba()
    iWidth, iHeight = fig.canvas.get_width_height()
    imgPil = Image.frombytes('RGBA', (iWidth, iHeight), daBuffer)
    return ImageTk.PhotoImage(imgPil, master=tkRoot)


def fnPlotCandidatePanel(lcNorm, dictCandidate, iNumber, iTotal):
    """Build a candidate flare figure (does not display it)."""
    daTime = np.array(lcNorm['time'].value, dtype=float)
    daFlux = np.array(lcNorm['flux'].value, dtype=float)
    dPeak = dictCandidate['dTimePeak']
    baWindow = (daTime >= dPeak - 0.05) & (daTime <= dPeak + 0.10)
    baCandidate = ((daTime >= dictCandidate['dTimeStart'])
                   & (daTime <= dictCandidate['dTimeStop']))
    plt.close('all')
    fig = plt.figure(figsize=(8, 5))
    plt.plot(daTime[baWindow] - dPeak, daFlux[baWindow], c='k')
    plt.scatter(daTime[baCandidate] - dPeak, daFlux[baCandidate],
                c=vplot.colors.pale_blue, zorder=5)
    plt.xlim(-0.05, 0.10)
    plt.xlabel('Time [days]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    sTitle = (f"Candidate {iNumber}/{iTotal}  |  "
              f"Sector {dictCandidate['iSectorNumber']}  |  "
              f"{dictCandidate['dPeakSigma']:.1f}\u03c3  |  "
              f"{dictCandidate['dDurationMinutes']:.0f} min")
    plt.title(sTitle, fontsize=14)
    plt.tight_layout()
    return fig


# ===== Interactive =====

def fsReadTerminalInput(sPrompt, tkRoot):
    """Read one line from stdin while keeping the Tk display alive."""
    sys.stdout.write(sPrompt)
    sys.stdout.flush()
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip().lower()
        tkRoot.update()
        tkRoot.after(100)


def fsPromptUserLabel(iNumber, iTotal, dictCandidate, tkRoot):
    """Prompt user to classify a candidate. Returns label string."""
    print(f"\nCandidate {iNumber}/{iTotal} | "
          f"Sector {dictCandidate['iSectorNumber']} | "
          f"Peak: {dictCandidate['dPeakSigma']:.1f} sigma | "
          f"Duration: {dictCandidate['dDurationMinutes']:.0f} min")
    dictLabels = {'f': 'flare', 'n': 'not_flare', 'u': 'uncertain',
                  'b': 'back', 'q': 'quit'}
    while True:
        sInput = fsReadTerminalInput(
            "  [f]lare  [n]ot-flare  [u]ncertain  "
            "[b]ack  [q]uit > ", tkRoot)
        if sInput in dictLabels:
            return dictLabels[sInput]
        print("  Invalid input. Enter f, n, u, b, or q.")


def fnRunInteractiveSession(listLightcurves, dictSession,
                            sOutputPath, iStartIndex=0):
    """Main interactive labeling loop with crash-safe saves."""
    tkRoot = tk.Tk()
    tkRoot.title("GJ 1132 Flare Candidates")
    tkLabel = tk.Label(tkRoot)
    tkLabel.pack()
    imgRef = None
    listCandidates = dictSession['listCandidates']
    iIndex = iStartIndex
    while iIndex < len(listCandidates):
        dictCandidate = listCandidates[iIndex]
        iSectorIdx = dictCandidate['iSectorIndex']
        lcNorm = listLightcurves[iSectorIdx].normalize()
        fig = fnPlotCandidatePanel(lcNorm, dictCandidate, iIndex + 1,
                                   len(listCandidates))
        imgRef = fnRenderFigureToTk(fig, tkRoot)
        tkLabel.configure(image=imgRef)
        tkRoot.update()
        plt.close(fig)
        sLabel = fsPromptUserLabel(iIndex + 1, len(listCandidates),
                                   dictCandidate, tkRoot)
        if sLabel == 'quit':
            break
        if sLabel == 'back':
            iIndex = max(0, iIndex - 1)
            continue
        dictCandidate['sLabel'] = sLabel
        fnSaveCandidatesToJson(dictSession, sOutputPath)
        iIndex += 1
    tkRoot.destroy()
    fnPrintLabelingSummary(listCandidates)


# ===== Persistence =====

def fdictCreateSession(listCandidates, dThreshold):
    """Create a new labeling session dictionary."""
    return {
        'dSigmaThreshold': dThreshold,
        'iWindowLengthFlatten': I_DEFAULT_WINDOW_LENGTH,
        'iTotalCandidates': len(listCandidates),
        'iLabeled': 0,
        'listCandidates': listCandidates,
    }


def fnSaveCandidatesToJson(dictSession, sOutputPath):
    """Save session state to JSON (called after each label)."""
    dictSession['iLabeled'] = fiCountLabeled(dictSession['listCandidates'])
    with open(sOutputPath, 'w') as fileHandle:
        json.dump(dictSession, fileHandle, indent=2)


def fdictLoadCandidatesFromJson(sInputPath):
    """Load a previous session from JSON."""
    with open(sInputPath, 'r') as fileHandle:
        return json.load(fileHandle)


def fiCountLabeled(listCandidates):
    """Return the number of candidates with a non-empty label."""
    return sum(1 for c in listCandidates if c.get('sLabel', ''))


# ===== Summary =====

def fnPrintLabelingSummary(listCandidates):
    """Print a summary of labeling results."""
    iFlares = sum(1 for c in listCandidates
                  if c.get('sLabel') == 'flare')
    iNotFlares = sum(1 for c in listCandidates
                     if c.get('sLabel') == 'not_flare')
    iUncertain = sum(1 for c in listCandidates
                     if c.get('sLabel') == 'uncertain')
    iUnlabeled = sum(1 for c in listCandidates if not c.get('sLabel'))
    print(f"\nLabeling summary:")
    print(f"  Flares:     {iFlares}")
    print(f"  Not flares: {iNotFlares}")
    print(f"  Uncertain:  {iUncertain}")
    print(f"  Unlabeled:  {iUnlabeled}")
    print(f"  Total:      {len(listCandidates)}")


# ===== Pipeline Bridge =====

def ftExtractFlareParameters(listCandidates):
    """Extract labeled flares in ftGetFlareParameters() format."""
    listFlares = [c for c in listCandidates if c['sLabel'] == 'flare']
    iaSectors = [c['iSectorIndex'] for c in listFlares]
    daTimeStart = np.array([c['dTimeStart'] for c in listFlares])
    daTimeStop = np.array([c['dTimeStop'] for c in listFlares])
    return iaSectors, daTimeStart, daTimeStop


# ===== Modes =====

def fnRunReviewMode(sInputPath):
    """Print summary of labeled candidates (non-interactive)."""
    dictSession = fdictLoadCandidatesFromJson(sInputPath)
    listCandidates = dictSession['listCandidates']
    print(f"Flare candidates: {sInputPath}")
    print(f"Threshold: {dictSession['dSigmaThreshold']} sigma")
    fnPrintLabelingSummary(listCandidates)
    listFlares = [c for c in listCandidates
                  if c.get('sLabel') == 'flare']
    for i, c in enumerate(listFlares):
        print(f"  Flare {i + 1}: Sector {c['iSectorNumber']}, "
              f"t={c['dTimePeak']:.4f}, {c['dPeakSigma']:.1f}\u03c3, "
              f"{c['dDurationMinutes']:.0f} min")


def fnRunScanMode(listLightcurves, dThreshold, sOutputPath):
    """Detect candidates and start interactive labeling."""
    print(f"Detecting candidates ({dThreshold} sigma threshold)...")
    listCandidates = flistDetectAllCandidates(listLightcurves, dThreshold)
    print(f"\nFound {len(listCandidates)} candidates total.\n")
    dictSession = fdictCreateSession(listCandidates, dThreshold)
    fnSaveCandidatesToJson(dictSession, sOutputPath)
    fnRunInteractiveSession(listLightcurves, dictSession, sOutputPath)


def fnRunResumeMode(listLightcurves, sInputPath):
    """Resume a partially-completed labeling session."""
    dictSession = fdictLoadCandidatesFromJson(sInputPath)
    listCandidates = dictSession['listCandidates']
    iStart = fiCountLabeled(listCandidates)
    print(f"Resuming: {iStart}/{len(listCandidates)} already labeled.\n")
    fnRunInteractiveSession(listLightcurves, dictSession, sInputPath,
                            iStart)


# ===== Entry Point =====

def main():
    """Parse arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Interactive flare candidate identification "
                    "for GJ 1132 TESS data.")
    parser.add_argument('--threshold', type=float,
                        default=D_DEFAULT_SIGMA_THRESHOLD,
                        help="Detection threshold in sigma "
                             f"(default: {D_DEFAULT_SIGMA_THRESHOLD})")
    parser.add_argument('--output', default='flare_candidates.json',
                        help="Output JSON path")
    parser.add_argument('--resume', metavar='PATH',
                        help="Resume labeling from a previous session.")
    parser.add_argument('--review', metavar='PATH',
                        help="Print summary of labeled candidates.")
    args = parser.parse_args()
    if args.review:
        fnRunReviewMode(args.review)
        return
    fnVerifyInteractiveDisplay()
    print("Loading TESS data for GJ 1132...")
    listLightcurves = flistDownloadTessData()
    print(f"Loaded {len(listLightcurves)} sectors.\n")
    if args.resume:
        fnRunResumeMode(listLightcurves, args.resume)
    else:
        fnRunScanMode(listLightcurves, args.threshold, args.output)


if __name__ == '__main__':
    main()
