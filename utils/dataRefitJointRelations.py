#!/usr/bin/env python3
"""
Joint hierarchical refit of the Engle rotation-age and activity-age relations.

One model per M dwarf sub-class fits BOTH relations simultaneously: the
rotation benchmarks (independent ages, latent true log-ages, split-normal
asymmetric age errors) calibrate the rotation relation, while the X-ray field
stars enter with LATENT ages informed by their rotation periods through that
same relation - so the gyrochronology-derived ages of the activity sample are
modeled rather than double-counted (their tabulated ages are never used).
Cluster-ensemble X-ray rows carry independent ages. Latent field-star and
ensemble ages are marginalized by Gauss-Hermite quadrature so the sampled
dimension stays near 50.

Both relations are continuous two-segment hinges y = a x + b + c max(x-d, 0).
Each carries a log-linear intrinsic-scatter law (Wallis-consistent split
normals for asymmetric errors; Kelly 2007 for the error framework):

    log sigma_int(x) = e + f * (x - pivot) / scale

The activity relation is fit in NATIVE log(L_X/L_bol); conversion to the
X-UV(5-1700 A) band happens downstream at prediction time via the re-derived
MUSCLES conversion (musclesConversion/conversionFit.json).

Sample composition follows the papers: M67 is excluded (gyro-circular age);
for the mid-late fit the young-track M4+ cluster rows are excluded because
Engle & Guinan (2023) calibrate the M2.5-6.5 young track on M2.5-3.5 stars
only. Halo subdwarfs are included or excluded via --subdwarfs, giving the
with/without sensitivity pair.

Convergence is gated on the integrated autocorrelation time of the 12
hyperparameters (chain length >= 50 tau; Foreman-Mackey et al. 2013).

Usage: python dataRefitJointRelations.py --sample midlate --subdwarfs exclude
"""

import argparse
import json
import os

import numpy as np
import emcee

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
I_HYPER_DIMENSIONS = 12
I_QUADRATURE_NODES = 40
I_WALKERS = 128
I_THIN_BY = 20
I_CHUNK_STORED = 200
I_MAX_STORED = 10000
D_CONVERGENCE_FACTOR = 50.0
D_MIN_SCATTER = 1e-3
D_MAX_SCATTER = 3.0

DICT_PUBLISHED = {
    "midlate": {"daRotation": [0.0561, -0.8900, -0.0521, 24.1888],
                "daActivity": [-0.2534, 28.6213 - 32.0, -3.8713, 0.3171],
                "daActivityRatio": [-0.2542, -3.2064, -3.1416, 0.3516],
                "tRotationBreakBounds": (5.0, 80.0),
                "tActivityBreakBounds": (-1.2, 1.2)},
    "early": {"daRotation": [0.0621, -1.0437, -0.0528, 23.4933],
              "daActivityRatio": [-0.8820, -3.8021, -0.7164, -0.3314],
              "tRotationBreakBounds": (5.0, 80.0),
              "tActivityBreakBounds": (-1.2, 1.2)},
}

# Young-track M4+ cluster rows excluded from the M2.5-6.5 rotation fit: the
# paper calibrates that fit's young track on M2.5-3.5 stars only.
LIST_MIDLATE_YOUNG_M4_ROWS = [
    ("Pleiades", "M4--~6.5V"), ("NGC 2516", "M4--6V"),
    ("Praesepe / Hyades", "M4--~6.5V"), ("NGC 752", "M4--6V"),
]

DA_NODES, DA_WEIGHTS = np.polynomial.hermite_e.hermegauss(I_QUADRATURE_NODES)
DA_LOG_WEIGHTS = np.log(DA_WEIGHTS / np.sqrt(2.0 * np.pi))


def fdaHinge(daCoefficients, daX):
    """Evaluate the continuous two-segment hinge y = a x + b + c max(x-d, 0)."""
    dA, dB, dC, dD = daCoefficients
    return dA * daX + dB + dC * np.where(daX >= dD, daX - dD, 0.0)


def fdaLocalSlope(daCoefficients, daX):
    """Return the hinge's local slope (a on segment 1, a+c on segment 2)."""
    dA, _, dC, dD = daCoefficients
    return dA + dC * (daX >= dD)


def fdaScatterLaw(dLogIntercept, dSlope, daX, dPivot, dScale):
    """Evaluate the log-linear intrinsic-scatter law sigma_int(x)."""
    return np.exp(dLogIntercept + dSlope * (daX - dPivot) / dScale)


def fdaSplitNormalLogPdf(daValue, daMean, daSigmaPlus, daSigmaMinus):
    """Log pdf of a split normal, continuous at the mode (Wallis 2014)."""
    daSigma = np.where(daValue >= daMean, daSigmaPlus, daSigmaMinus)
    daNormalization = np.log(np.sqrt(2.0 / np.pi) / (daSigmaPlus + daSigmaMinus))
    return daNormalization - 0.5 * ((daValue - daMean) / daSigma) ** 2


def ftBuildLogAgeErrors(daAge, daPlus, daMinus):
    """Return observed log-age and its asymmetric log-space uncertainties."""
    daTauObserved = np.log10(daAge)
    daSigmaPlus = np.log10(daAge + daPlus) - daTauObserved
    daSigmaMinus = daTauObserved - np.log10(np.clip(daAge - daMinus, 1e-3, None))
    return daTauObserved, daSigmaPlus, daSigmaMinus


def fbIsExcludedMidlateYoungRow(dictRecord, sSample):
    """Return True for a young-track M4+ cluster row in the mid-late fit."""
    if sSample != "midlate":
        return False
    return (dictRecord["sName"], dictRecord["sSptype"]) in \
        LIST_MIDLATE_YOUNG_M4_ROWS


def flistFilterRecords(listRecords, sSample, bIncludeSubdwarfs):
    """Apply the documented sample-composition rules."""
    listKept = []
    for dictRecord in listRecords:
        if dictRecord["bExcludedFromPaperFits"]:
            continue
        if dictRecord["bSubdwarf"] and not bIncludeSubdwarfs:
            continue
        if fbIsExcludedMidlateYoungRow(dictRecord, sSample):
            continue
        listKept.append(dictRecord)
    return listKept


def fdSymmetrizedProtError(dictRotation, dMedianRelative):
    """Return a symmetric Prot error, imputing the median relative error."""
    if dictRotation["dPlus"] is None or dictRotation["dMinus"] is None:
        return dMedianRelative * dictRotation["dProtDays"]
    return 0.5 * (dictRotation["dPlus"] + dictRotation["dMinus"])


def fdictAssembleArrays(listRecords):
    """Split records into benchmark, field, and ensemble numpy blocks."""
    listBench = [d for d in listRecords if d["sAgeProvenance"] == "independent"
                 and d["dictRotation"] is not None]
    listField = [d for d in listRecords if d["sAgeProvenance"] == "latent"]
    listEnsemble = [d for d in listRecords if d["bClusterEnsemble"]]
    daRelative = np.array(
        [0.5 * (d["dictRotation"]["dPlus"] + d["dictRotation"]["dMinus"])
         / d["dictRotation"]["dProtDays"] for d in listBench
         if d["dictRotation"]["dPlus"] is not None])
    dMedianRelative = float(np.median(daRelative))
    return {"listBench": listBench, "listField": listField,
            "listEnsemble": listEnsemble, "dMedianRelative": dMedianRelative}


def fdictBenchmarkBlock(listBench, dMedianRelative):
    """Build the benchmark arrays, including the optional X-ray attachments."""
    daAge = np.array([d["dictAgeConstraint"]["dValue"] for d in listBench])
    daPlus = np.array([d["dictAgeConstraint"]["dPlus"] for d in listBench])
    daMinus = np.array([d["dictAgeConstraint"]["dMinus"] for d in listBench])
    daTauObs, daSigPlus, daSigMinus = ftBuildLogAgeErrors(daAge, daPlus, daMinus)
    baHasXray = np.array([d["dictXray"] is not None for d in listBench])
    return {
        "daProt": np.array([d["dictRotation"]["dProtDays"] for d in listBench]),
        "daSigProt": np.array([fdSymmetrizedProtError(d["dictRotation"],
                               dMedianRelative) for d in listBench]),
        "daTauObs": daTauObs, "daSigPlus": daSigPlus, "daSigMinus": daSigMinus,
        "baHasXray": baHasXray,
        "daY": np.array([d["dictXray"]["dLogLxLbol"] if d["dictXray"] else 0.0
                         for d in listBench]),
        "daYErr": np.array([d["dictXray"]["dError"] if d["dictXray"] else 1.0
                            for d in listBench]),
    }


def fdictFieldBlock(listField):
    """Build the latent-age field-star arrays (Prot plus X-ray)."""
    return {
        "daProt": np.array([d["dictRotation"]["dProtDays"] for d in listField]),
        "daSigProt": np.array([d["dictRotation"]["dPlus"] for d in listField]),
        "daY": np.array([d["dictXray"]["dLogLxLbol"] for d in listField]),
        "daYErr": np.array([d["dictXray"]["dError"] for d in listField]),
    }


def fdictEnsembleBlock(listEnsemble):
    """Build the cluster-ensemble arrays (independent log-space ages)."""
    return {
        "daTau": np.array([d["dictAgeConstraint"]["dValue"]
                           for d in listEnsemble]),
        "daSigTau": np.array([d["dictAgeConstraint"]["dError"]
                              for d in listEnsemble]),
        "daY": np.array([d["dictXray"]["dLogLxLbol"] for d in listEnsemble]),
        "daYErr": np.array([d["dictXray"]["dError"] for d in listEnsemble]),
    }


def fdictBuildData(sSample, bIncludeSubdwarfs):
    """Load, filter, and assemble all model inputs for one fit variant."""
    sFilename = ("jointSampleMidLate.json" if sSample == "midlate"
                 else "jointSampleEarly.json")
    with open(os.path.join(S_DIRECTORY, sFilename)) as fileHandle:
        listRecords = json.load(fileHandle)
    listKept = flistFilterRecords(listRecords, sSample, bIncludeSubdwarfs)
    dictSplit = fdictAssembleArrays(listKept)
    dictBench = fdictBenchmarkBlock(dictSplit["listBench"],
                                    dictSplit["dMedianRelative"])
    dictData = {"dictBench": dictBench,
                "dictField": fdictFieldBlock(dictSplit["listField"]),
                "dictEnsemble": fdictEnsembleBlock(dictSplit["listEnsemble"]),
                "iBenchmarks": len(dictSplit["listBench"]),
                "saBenchNames": [d["sName"] for d in dictSplit["listBench"]]}
    dictData["dPivotProt"] = float(np.mean(dictBench["daProt"]))
    dictData["dScaleProt"] = float(np.std(dictBench["daProt"]))
    dictData["dPivotTau"] = float(np.mean(dictBench["daTauObs"]))
    dictData["dScaleTau"] = max(float(np.std(dictBench["daTauObs"])), 0.3)
    return dictData


def fdHyperLogPrior(daTheta, dictBounds):
    """Weakly-informative prior over the 12 hyperparameters."""
    daCr, daCx = daTheta[0:4], daTheta[6:10]
    dEr, dFr, dEx, dFx = daTheta[4], daTheta[5], daTheta[10], daTheta[11]
    if not (dictBounds["tRotationBreakBounds"][0] < daCr[3]
            < dictBounds["tRotationBreakBounds"][1]):
        return -np.inf
    if not (dictBounds["tActivityBreakBounds"][0] < daCx[3]
            < dictBounds["tActivityBreakBounds"][1]):
        return -np.inf
    dLogPrior = -0.5 * ((daCr[0] / 0.1) ** 2 + (daCr[1] / 5.0) ** 2
                        + (daCr[2] / 0.1) ** 2)
    dLogPrior += -0.5 * ((daCx[0] / 1.0) ** 2 + ((daCx[1] + 3.5) / 2.0) ** 2
                         + (daCx[2] / 3.0) ** 2)
    dLogPrior += -0.5 * (((dEr - np.log(0.15)) / 1.5) ** 2 + (dFr / 2.0) ** 2)
    dLogPrior += -0.5 * (((dEx - np.log(0.30)) / 1.5) ** 2 + (dFx / 2.0) ** 2)
    return dLogPrior


def fdaLatentAgeSpread(daCr, dEr, dFr, daProt, daSigProt, dictData):
    """Return the total spread of latent log-age given Prot (scatter + slope)."""
    daScatter = fdaScatterLaw(dEr, dFr, daProt, dictData["dPivotProt"],
                              dictData["dScaleProt"])
    if np.any(daScatter > D_MAX_SCATTER) or np.any(daScatter < D_MIN_SCATTER):
        return None
    daSlope = fdaLocalSlope(daCr, daProt)
    return np.sqrt(daScatter ** 2 + (daSlope * daSigProt) ** 2)


def fdaQuadratureActivityLogLike(daMu, daSpread, daY, daYErr, daCx, dEx, dFx,
                                 dictData):
    """Marginal log-likelihood of X-ray data over latent ages by quadrature."""
    daTauGrid = daMu[:, None] + daSpread[:, None] * DA_NODES[None, :]
    daModelY = fdaHinge(daCx, daTauGrid)
    daScatter = fdaScatterLaw(dEx, dFx, daTauGrid, dictData["dPivotTau"],
                              dictData["dScaleTau"])
    if np.any(daScatter > D_MAX_SCATTER):
        return None
    daVariance = daYErr[:, None] ** 2 + daScatter ** 2
    daLogPdf = (-0.5 * (daY[:, None] - daModelY) ** 2 / daVariance
                - 0.5 * np.log(2.0 * np.pi * daVariance))
    daTerms = DA_LOG_WEIGHTS[None, :] + daLogPdf
    daMax = np.max(daTerms, axis=1)
    return daMax + np.log(np.sum(np.exp(daTerms - daMax[:, None]), axis=1))


def fdBenchmarkLogLike(daTheta, dictData):
    """Rotation-relation likelihood of the benchmarks with latent ages."""
    daCr, dEr, dFr = daTheta[0:4], daTheta[4], daTheta[5]
    daLatent = daTheta[I_HYPER_DIMENSIONS:]
    dictBench = dictData["dictBench"]
    daSpread = fdaLatentAgeSpread(daCr, dEr, dFr, dictBench["daProt"],
                                  dictBench["daSigProt"], dictData)
    if daSpread is None:
        return -np.inf
    daMu = fdaHinge(daCr, dictBench["daProt"])
    dLogLike = np.sum(-0.5 * ((daLatent - daMu) / daSpread) ** 2
                      - np.log(daSpread))
    dLogLike += np.sum(fdaSplitNormalLogPdf(
        dictBench["daTauObs"], daLatent,
        dictBench["daSigPlus"], dictBench["daSigMinus"]))
    return dLogLike


def fdBenchmarkXrayLogLike(daTheta, dictData):
    """Activity likelihood of benchmarks that also carry X-ray data."""
    daCx, dEx, dFx = daTheta[6:10], daTheta[10], daTheta[11]
    dictBench = dictData["dictBench"]
    baMask = dictBench["baHasXray"]
    if not np.any(baMask):
        return 0.0
    daLatent = daTheta[I_HYPER_DIMENSIONS:][baMask]
    daScatter = fdaScatterLaw(dEx, dFx, daLatent, dictData["dPivotTau"],
                              dictData["dScaleTau"])
    if np.any(daScatter > D_MAX_SCATTER):
        return -np.inf
    daVariance = dictBench["daYErr"][baMask] ** 2 + daScatter ** 2
    daResidual = dictBench["daY"][baMask] - fdaHinge(daCx, daLatent)
    return float(np.sum(-0.5 * daResidual ** 2 / daVariance
                        - 0.5 * np.log(2.0 * np.pi * daVariance)))


def fdJointLogPosterior(daTheta, dictData, dictBounds):
    """Full joint log-posterior over hyperparameters and benchmark latents."""
    dLogPrior = fdHyperLogPrior(daTheta, dictBounds)
    daLatent = daTheta[I_HYPER_DIMENSIONS:]
    if not np.isfinite(dLogPrior) or np.any(daLatent < -2.0) \
            or np.any(daLatent > 1.3):
        return -np.inf
    dTotal = dLogPrior + fdBenchmarkLogLike(daTheta, dictData)
    if not np.isfinite(dTotal):
        return -np.inf
    dTotal += fdBenchmarkXrayLogLike(daTheta, dictData)
    dictField = dictData["dictField"]
    daSpread = fdaLatentAgeSpread(daTheta[0:4], daTheta[4], daTheta[5],
                                  dictField["daProt"], dictField["daSigProt"],
                                  dictData)
    if daSpread is None:
        return -np.inf
    daMu = fdaHinge(daTheta[0:4], dictField["daProt"])
    daFieldLike = fdaQuadratureActivityLogLike(
        daMu, daSpread, dictField["daY"], dictField["daYErr"],
        daTheta[6:10], daTheta[10], daTheta[11], dictData)
    dictEnsemble = dictData["dictEnsemble"]
    daEnsembleLike = fdaQuadratureActivityLogLike(
        dictEnsemble["daTau"], dictEnsemble["daSigTau"], dictEnsemble["daY"],
        dictEnsemble["daYErr"], daTheta[6:10], daTheta[10], daTheta[11],
        dictData)
    if daFieldLike is None or daEnsembleLike is None:
        return -np.inf
    return dTotal + np.sum(daFieldLike) + np.sum(daEnsembleLike)


def fdaInitialGuess(sSample, dictData):
    """Return the initial parameter vector from the published coefficients."""
    dictPub = DICT_PUBLISHED[sSample]
    daActivity = dictPub["daActivityRatio"]
    daHyper = np.concatenate([
        dictPub["daRotation"], [np.log(0.15), 0.0],
        daActivity, [np.log(0.30), 0.0]])
    return np.concatenate([daHyper, dictData["dictBench"]["daTauObs"]])


LIST_MOVES = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


def ftRunConvergedSampler(fdLogPosterior, daInitial, tArgs, iSeed):
    """Run emcee (DE moves, thinned storage) until the 50-tau gate passes.

    Storing every I_THIN_BY-th proposal bounds memory; the convergence
    inequality is invariant under thinning. DE moves mix far better than the
    stretch move in this correlated hierarchical space.
    """
    np.random.seed(iSeed)
    iDimensions = len(daInitial)
    daStart = daInitial + 1e-4 * np.random.randn(I_WALKERS, iDimensions)
    sampler = emcee.EnsembleSampler(I_WALKERS, iDimensions, fdLogPosterior,
                                    args=tArgs, moves=LIST_MOVES)
    state, iStored, bConverged, dTauStored = daStart, 0, False, np.inf
    while iStored < I_MAX_STORED and not bConverged:
        state = sampler.run_mcmc(state, I_CHUNK_STORED, thin_by=I_THIN_BY,
                                 progress=False)
        iStored += I_CHUNK_STORED
        dTauStored = float(np.max(
            sampler.get_autocorr_time(tol=0)[:I_HYPER_DIMENSIONS]))
        bConverged = iStored > D_CONVERGENCE_FACTOR * dTauStored
        print(f"  proposals={iStored * I_THIN_BY} "
              f"tau={dTauStored * I_THIN_BY:.0f} converged={bConverged}",
              flush=True)
    return sampler, iStored, bConverged, dTauStored


def fdictSummarizeFit(sampler, iSteps, bConverged, dTauMax, dictData, sSample):
    """Build the posterior summary dictionary for one fit variant."""
    iDiscard = int(min(5 * dTauMax, iSteps // 2))
    iThin = max(1, int(dTauMax / 2))
    daFlat = sampler.get_chain(discard=iDiscard, thin=iThin, flat=True)
    saNames = ["a_rot", "b_rot", "c_rot", "d_rot", "e_rot", "f_rot",
               "a_act", "b_act", "c_act", "d_act", "e_act", "f_act"]
    dictSummary = {"sSample": sSample, "bConverged": bConverged,
                   "iSteps": iSteps * I_THIN_BY,
                   "dTauMax": dTauMax * I_THIN_BY,
                   "iThinBy": I_THIN_BY,
                   "iPosteriorDraws": int(daFlat.shape[0]),
                   "dPivotProt": dictData["dPivotProt"],
                   "dScaleProt": dictData["dScaleProt"],
                   "dPivotTau": dictData["dPivotTau"],
                   "dScaleTau": dictData["dScaleTau"]}
    for iIndex, sName in enumerate(saNames):
        daColumn = daFlat[:, iIndex]
        dictSummary[sName] = {"mean": float(np.mean(daColumn)),
                              "std": float(np.std(daColumn)),
                              "ci68": [float(np.percentile(daColumn, 16)),
                                       float(np.percentile(daColumn, 84))]}
    return dictSummary, daFlat


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Joint hierarchical refit of the Engle relations.")
    parser.add_argument("--sample", required=True,
                        choices=["midlate", "early"])
    parser.add_argument("--subdwarfs", required=True,
                        choices=["include", "exclude"])
    return parser.parse_args()


def main():
    """Run one fit variant end to end and persist its posterior products."""
    args = ftParseArguments()
    bInclude = args.subdwarfs == "include"
    sTag = f"{args.sample}_{'withsd' if bInclude else 'nosd'}"
    dictData = fdictBuildData(args.sample, bInclude)
    print(f"[{sTag}] benchmarks={dictData['iBenchmarks']} "
          f"field={len(dictData['dictField']['daY'])} "
          f"ensembles={len(dictData['dictEnsemble']['daY'])}", flush=True)
    iSeed = {"midlate_withsd": 42, "midlate_nosd": 43,
             "early_withsd": 44, "early_nosd": 45}[sTag]
    sampler, iSteps, bConverged, dTauMax = ftRunConvergedSampler(
        fdJointLogPosterior, fdaInitialGuess(args.sample, dictData),
        (dictData, DICT_PUBLISHED[args.sample]), iSeed)
    dictSummary, daFlat = fdictSummarizeFit(
        sampler, iSteps, bConverged, dTauMax, dictData, args.sample)
    np.save(os.path.join(S_DIRECTORY, f"jointChain_{sTag}.npy"),
            daFlat[:, :I_HYPER_DIMENSIONS])
    with open(os.path.join(S_DIRECTORY, f"jointFitSummary_{sTag}.json"),
              "w") as fileHandle:
        json.dump(dictSummary, fileHandle, indent=2)
    print(f"[{sTag}] done: converged={bConverged} tau={dTauMax:.1f} "
          f"draws={dictSummary['iPosteriorDraws']}", flush=True)


if __name__ == "__main__":
    main()
