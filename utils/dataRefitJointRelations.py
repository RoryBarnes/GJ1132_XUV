#!/usr/bin/env python3
"""
Joint hierarchical refit of the Engle rotation-age and activity-age relations.

One model per M dwarf sub-class fits BOTH relations simultaneously: the
rotation benchmarks (independent ages, latent true log-ages, split-normal
asymmetric age errors) calibrate the rotation relation, while the X-ray field
stars enter with LATENT ages informed by their rotation periods through that
same relation - so the gyrochronology-derived ages of the activity sample are
modeled rather than double-counted (their tabulated ages are never used).
Cluster-ensemble X-ray rows carry independent ages. EVERY latent age -
benchmark, field-star and ensemble - is marginalized by Gauss-Hermite
quadrature, so only the 12 hyperparameters are sampled. Sampling the 34
benchmark latents jointly with the scatter (the centered parameterization)
creates a funnel (Neal 2003, Ann. Stat. 31, 705) into which emcee walkers
collapse from any start, underestimating the rotation scatter roughly
tenfold; marginalizing removes it (Papaspiliopoulos, Roberts & Skold 2007,
Stat. Sci. 22, 59). Each benchmark integral uses nodes centered on a
Gaussian approximation to that star's latent-age posterior.

Both relations are continuous two-segment hinges y = a x + b + c max(x-d, 0).
Each carries a log-linear intrinsic-scatter law (Wallis-consistent split
normals for asymmetric errors; Kelly 2007 for the error framework):

    log sigma_int(x) = e + f * (x - pivot) / scale

The activity relation is fit in NATIVE log(L_X/L_bol); conversion to the
X-UV(5-1700 A) band happens downstream at prediction time via the re-derived
MUSCLES conversion (the MusclesBandConversion step's conversionFit.json).

Sample composition follows the papers: M67 is excluded (gyro-circular age);
for the mid-late fit the young-track M4+ cluster rows are excluded because
Engle & Guinan (2023) calibrate the M2.5-6.5 young track on M2.5-3.5 stars
only. Halo subdwarfs are included or excluded via --subdwarfs, giving the
with/without sensitivity pair.

The posterior is sampled twice with independent algorithms (utils/
sampleJointPosterior.py): dynesty nested sampling (Speagle 2020), whose chain
is the step's primary product, and four emcee runs from dispersed starts,
gated on 50 integrated autocorrelation times per run (Foreman-Mackey et al.
2013) and split-R-hat (Gelman et al. 2013, BDA3, Section 11.4), as a
cross-check.

Usage: python dataRefitJointRelations.py --sample midlate --subdwarfs exclude
       --joint-sample jointSampleMidLate.json --output-directory .
"""

import argparse
import json
import os

import numpy as np

I_HYPER_DIMENSIONS = 12
I_QUADRATURE_NODES = 40
D_PROPOSAL_WIDENING = 1.5
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


def fdictBuildData(sJointSamplePath, sSample, bIncludeSubdwarfs):
    """Load, filter, and assemble all model inputs for one fit variant."""
    with open(sJointSamplePath) as fileHandle:
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


def flistPriorSpecification(dictBounds):
    """Return each hyperparameter's prior as (kind, location/low, scale/high).

    The single definition both fdHyperLogPrior and the nested sampler's prior
    transform read, so the two cannot drift apart.
    """
    return [("normal", 0.0, 0.1), ("normal", 0.0, 5.0), ("normal", 0.0, 0.1),
            ("uniform", *dictBounds["tRotationBreakBounds"]),
            ("normal", np.log(0.15), 1.5), ("normal", 0.0, 2.0),
            ("normal", 0.0, 1.0), ("normal", -3.5, 2.0), ("normal", 0.0, 3.0),
            ("uniform", *dictBounds["tActivityBreakBounds"]),
            ("normal", np.log(0.30), 1.5), ("normal", 0.0, 2.0)]


def fdHyperLogPrior(daTheta, dictBounds):
    """Weakly-informative prior over the 12 hyperparameters (unnormalized)."""
    dLogPrior = 0.0
    for dValue, (sKind, dFirst, dSecond) in zip(
            daTheta, flistPriorSpecification(dictBounds)):
        if sKind == "uniform" and not dFirst < dValue < dSecond:
            return -np.inf
        if sKind == "normal":
            dLogPrior += -0.5 * ((dValue - dFirst) / dSecond) ** 2
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


def ftBenchmarkLatentPrior(daTheta, dictData):
    """Return each benchmark's latent log-age prior mean and spread, or None."""
    dictBench = dictData["dictBench"]
    daSpread = fdaLatentAgeSpread(daTheta[0:4], daTheta[4], daTheta[5],
                                  dictBench["daProt"], dictBench["daSigProt"],
                                  dictData)
    if daSpread is None:
        return None
    return fdaHinge(daTheta[0:4], dictBench["daProt"]), daSpread


def fdaBenchmarkLogIntegrand(daTau, iStar, daTheta, dictData, tLatentPrior):
    """Log of latent-age prior x age likelihood x optional X-ray likelihood."""
    dictBench = dictData["dictBench"]
    daMu, daSpread = tLatentPrior
    daLog = (-0.5 * ((daTau - daMu[iStar]) / daSpread[iStar]) ** 2
             - np.log(daSpread[iStar] * np.sqrt(2.0 * np.pi)))
    daLog = daLog + fdaSplitNormalLogPdf(
        dictBench["daTauObs"][iStar], daTau,
        dictBench["daSigPlus"][iStar], dictBench["daSigMinus"][iStar])
    if dictBench["baHasXray"][iStar]:
        daScatter = fdaScatterLaw(daTheta[10], daTheta[11], daTau,
                                  dictData["dPivotTau"], dictData["dScaleTau"])
        daVariance = dictBench["daYErr"][iStar] ** 2 + daScatter ** 2
        daResidual = dictBench["daY"][iStar] - fdaHinge(daTheta[6:10], daTau)
        daLog = daLog - 0.5 * (daResidual ** 2 / daVariance
                               + np.log(2.0 * np.pi * daVariance))
    return daLog


def ftLatentPosteriorApproximation(iStar, dictBench, tLatentPrior):
    """Gaussian approximation (center, widened width) to one latent posterior."""
    daMu, daSpread = tLatentPrior
    dSigmaObserved = 0.5 * (dictBench["daSigPlus"][iStar]
                            + dictBench["daSigMinus"][iStar])
    dVariance = 1.0 / (1.0 / daSpread[iStar] ** 2 + 1.0 / dSigmaObserved ** 2)
    dCenter = dVariance * (daMu[iStar] / daSpread[iStar] ** 2
                           + dictBench["daTauObs"][iStar] / dSigmaObserved ** 2)
    return dCenter, D_PROPOSAL_WIDENING * np.sqrt(dVariance)


def fdLogBenchmarkIntegral(iStar, daTheta, dictData, tLatentPrior):
    """Marginalize one benchmark's latent log-age by importance Gauss-Hermite."""
    dCenter, dWidth = ftLatentPosteriorApproximation(
        iStar, dictData["dictBench"], tLatentPrior)
    daTau = dCenter + dWidth * DA_NODES
    daTerms = (np.log(DA_WEIGHTS) + 0.5 * DA_NODES ** 2
               + fdaBenchmarkLogIntegrand(daTau, iStar, daTheta, dictData,
                                          tLatentPrior))
    dMax = np.max(daTerms)
    return float(np.log(dWidth) + dMax + np.log(np.sum(np.exp(daTerms - dMax))))


def fdBenchmarkMarginalLogLike(daTheta, dictData):
    """Benchmark likelihood with every latent true log-age marginalized."""
    tLatentPrior = ftBenchmarkLatentPrior(daTheta, dictData)
    if tLatentPrior is None:
        return -np.inf
    return sum(fdLogBenchmarkIntegral(iStar, daTheta, dictData, tLatentPrior)
               for iStar in range(dictData["iBenchmarks"]))


def fdNonBenchmarkLogLike(daTheta, dictData):
    """Field-star and cluster-ensemble activity likelihood by quadrature."""
    dictField, dictEnsemble = dictData["dictField"], dictData["dictEnsemble"]
    daSpread = fdaLatentAgeSpread(daTheta[0:4], daTheta[4], daTheta[5],
                                  dictField["daProt"], dictField["daSigProt"],
                                  dictData)
    if daSpread is None:
        return -np.inf
    daFieldLike = fdaQuadratureActivityLogLike(
        fdaHinge(daTheta[0:4], dictField["daProt"]), daSpread, dictField["daY"],
        dictField["daYErr"], daTheta[6:10], daTheta[10], daTheta[11], dictData)
    daEnsembleLike = fdaQuadratureActivityLogLike(
        dictEnsemble["daTau"], dictEnsemble["daSigTau"], dictEnsemble["daY"],
        dictEnsemble["daYErr"], daTheta[6:10], daTheta[10], daTheta[11],
        dictData)
    if daFieldLike is None or daEnsembleLike is None:
        return -np.inf
    return float(np.sum(daFieldLike) + np.sum(daEnsembleLike))


def fdLogLikelihood(daTheta, dictData):
    """Full marginalized log-likelihood of the 12 hyperparameters."""
    dBenchmark = fdBenchmarkMarginalLogLike(daTheta, dictData)
    if not np.isfinite(dBenchmark):
        return -np.inf
    return dBenchmark + fdNonBenchmarkLogLike(daTheta, dictData)


def fdJointLogPosterior(daTheta, dictData, dictBounds):
    """Log-posterior of the 12 hyperparameters (latent ages marginalized)."""
    dLogPrior = fdHyperLogPrior(daTheta, dictBounds)
    if not np.isfinite(dLogPrior):
        return -np.inf
    dLogLikelihood = fdLogLikelihood(daTheta, dictData)
    return dLogPrior + dLogLikelihood if np.isfinite(dLogLikelihood) else -np.inf


def fdictSummarizeChain(daChain, dictData, sSample):
    """Build the per-parameter posterior summary for one chain."""
    saNames = ["a_rot", "b_rot", "c_rot", "d_rot", "e_rot", "f_rot",
               "a_act", "b_act", "c_act", "d_act", "e_act", "f_act"]
    dictSummary = {"sSample": sSample, "iPosteriorDraws": int(daChain.shape[0]),
                   "dPivotProt": dictData["dPivotProt"],
                   "dScaleProt": dictData["dScaleProt"],
                   "dPivotTau": dictData["dPivotTau"],
                   "dScaleTau": dictData["dScaleTau"]}
    for iIndex, sName in enumerate(saNames):
        daColumn = daChain[:, iIndex]
        dictSummary[sName] = {"mean": float(np.mean(daColumn)),
                              "std": float(np.std(daColumn)),
                              "ci68": [float(np.percentile(daColumn, 16)),
                                       float(np.percentile(daColumn, 84))]}
    return dictSummary


def ftParseArguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Joint hierarchical refit of the Engle relations.")
    parser.add_argument("--sample", required=True,
                        choices=["midlate", "early"])
    parser.add_argument("--subdwarfs", required=True,
                        choices=["include", "exclude"])
    parser.add_argument("--joint-sample", required=True,
                        help="Path to the fit-ready joint sample JSON.")
    parser.add_argument("--output-directory", default=".",
                        help="Directory receiving the chain and summary.")
    return parser.parse_args()


def fnSaveProducts(sDirectory, sTag, daNested, daEmcee, dictSummary):
    """Write the primary (nested) chain, the emcee chain, and the summary."""
    np.save(os.path.join(sDirectory, f"jointChain_{sTag}.npy"), daNested)
    np.save(os.path.join(sDirectory, f"jointChainEmcee_{sTag}.npy"), daEmcee)
    with open(os.path.join(sDirectory, f"jointFitSummary_{sTag}.json"),
              "w") as fileHandle:
        json.dump(dictSummary, fileHandle, indent=2)


def main():
    """Run one fit variant with both samplers and persist the products."""
    from sampleJointPosterior import fdictRunNestedSampler, fdictRunEmceeRuns
    args = ftParseArguments()
    bInclude = args.subdwarfs == "include"
    sTag = f"{args.sample}_{'withsd' if bInclude else 'nosd'}"
    dictData = fdictBuildData(args.joint_sample, args.sample, bInclude)
    print(f"[{sTag}] benchmarks={dictData['iBenchmarks']} "
          f"field={len(dictData['dictField']['daY'])} "
          f"ensembles={len(dictData['dictEnsemble']['daY'])}", flush=True)
    iSeed = {"midlate_withsd": 42, "midlate_nosd": 43,
             "early_withsd": 44, "early_nosd": 45}[sTag]
    dictNested = fdictRunNestedSampler(dictData, args.sample, iSeed)
    dictEmcee = fdictRunEmceeRuns(dictData, args.sample, iSeed)
    daNested = dictNested.pop("daChain")
    dictSummary = fdictSummarizeChain(daNested, dictData, args.sample)
    dictSummary.update({"dictNested": dictNested, "dictEmcee": dictEmcee["dictDiagnostics"]})
    print(f"[{sTag}] done: nested draws={dictSummary['iPosteriorDraws']} "
          f"emcee converged={dictEmcee['dictDiagnostics']['bConverged']}", flush=True)
    fnSaveProducts(args.output_directory, sTag, daNested, dictEmcee["daChain"], dictSummary)


if __name__ == "__main__":
    main()
