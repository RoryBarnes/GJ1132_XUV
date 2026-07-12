#!/usr/bin/env python3
"""
Hierarchical Bayesian refit of the two Engle EMD relations from per-star data.

Refits the age-rotation relation (Engle & Guinan 2023, M4-6.5 benchmarks) and
the X-UV(5-1700A)/L_bol activity-age relation (Engle 2024, mid-late M X-ray
sample, reconstructed onto the X-UV band). Both relations are continuous
two-segment (broken/hinge) lines. The model carries measurement errors plus a
segment-dependent intrinsic scatter (sigma_int) inferred as a hyperparameter -
a quantity neither paper reports. Sampling uses emcee (NUTS packages absent).

References: Kelly (2007) ApJ 665, 1489 (measurement-error + intrinsic-scatter
regression); Engle & Guinan (2023) ApJL 954, L50; Engle (2024) ApJ 960, 62.
"""

import csv
import json
import os

import numpy as np
import emcee

I_SEED = 42
S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Published central EMD coefficients (manuscript Table tab:emd / Engle papers).
DA_XUV_PUBLISHED = np.array([-0.1456, -2.8876, -1.8187, 0.3545])
DA_LXBOL_PUBLISHED = np.array([-0.2542, -3.2064, -3.1416, 0.3516])
DA_ROT_PUBLISHED = np.array([0.0251, -0.1615, -0.0212, 25.4500])


def fdaHinge(daCoefficients, daX):
    """Evaluate the continuous two-segment hinge y = a x + b + c max(x-d, 0)."""
    dA, dB, dC, dD = daCoefficients
    return dA * daX + dB + dC * np.where(daX >= dD, daX - dD, 0.0)


def fdaLocalSlope(daCoefficients, daX):
    """Return the local slope of the hinge (a on segment 1, a+c on segment 2)."""
    dA, _, dC, dD = daCoefficients
    return dA + dC * (daX >= dD)


def fdSplitNormalLogPdf(daValue, daMean, daSigmaPlus, daSigmaMinus):
    """Log pdf of a split-normal (asymmetric Gaussian) evaluated elementwise."""
    daSigma = np.where(daValue >= daMean, daSigmaPlus, daSigmaMinus)
    daNormalization = np.log(np.sqrt(2.0 / np.pi) / (daSigmaPlus + daSigmaMinus))
    return daNormalization - 0.5 * ((daValue - daMean) / daSigma) ** 2


def fdSigmaLogPrior(dSigma, dScale):
    """Half-normal prior on a positive scatter, expressed in log-sigma space."""
    if dSigma <= 1e-3 or dSigma >= 3.0:
        return -np.inf
    return -0.5 * (dSigma / dScale) ** 2 + np.log(dSigma)


def fdSegmentSigma(dSigmaYoung, dSigmaOld, daX, dBreak):
    """Return the per-point intrinsic scatter given the segment membership."""
    return np.where(daX >= dBreak, dSigmaOld, dSigmaYoung)


def ftLoadRotationBenchmarks():
    """Return (Prot, age, age_plus, age_minus) arrays for M4-6.5 benchmarks."""
    listRows = list(csv.DictReader(
        open(os.path.join(S_DIRECTORY, "rotationBenchmarks.csv"))))
    listSelected = [dictRow for dictRow in listRows
                    if fbIsM4ToM6Point5(dictRow["sptype"])]
    daProt = np.array([float(r["prot_days"]) for r in listSelected])
    daAge = np.array([float(r["age_gyr"]) for r in listSelected])
    daPlus = np.array([float(r["age_plus"]) for r in listSelected])
    daMinus = np.array([float(r["age_minus"]) for r in listSelected])
    return daProt, daAge, daPlus, daMinus


def fbIsM4ToM6Point5(sSpectralType):
    """Return True for a dwarf (class V) M4.0-M6.5, excluding subdwarfs."""
    if "sd" in sSpectralType:
        return False
    import re
    matchType = re.search(r"M(\d+\.?\d*)", sSpectralType)
    if not matchType:
        return False
    dSubtype = float(matchType.group(1))
    return 4.0 <= dSubtype <= 6.5


def ftBuildRotationLogAgeErrors(daAge, daPlus, daMinus):
    """Return observed log-age and its asymmetric log-space uncertainties."""
    daTauObserved = np.log10(daAge)
    daSigmaPlus = np.log10(daAge + daPlus) - daTauObserved
    daSigmaMinus = daTauObserved - np.log10(np.clip(daAge - daMinus, 1e-3, None))
    return daTauObserved, daSigmaPlus, daSigmaMinus


def fdRotationLogPosterior(daParameters, daProt, daTauObs, daSigPlus, daSigMinus):
    """Log posterior for the rotation relation with latent per-star log-ages."""
    daCoefficients = daParameters[:4]
    dSigmaYoung, dSigmaOld = np.exp(daParameters[4]), np.exp(daParameters[5])
    daLatentTau = daParameters[6:]
    dLogPrior = fdRotationLogPrior(daCoefficients, dSigmaYoung, dSigmaOld,
                                   daLatentTau)
    if not np.isfinite(dLogPrior):
        return -np.inf
    daModelTau = fdaHinge(daCoefficients, daProt)
    daSigmaSegment = fdSegmentSigma(dSigmaYoung, dSigmaOld, daProt,
                                    daCoefficients[3])
    dLogScatter = np.sum(-0.5 * ((daLatentTau - daModelTau) / daSigmaSegment) ** 2
                         - np.log(daSigmaSegment))
    dLogMeasurement = np.sum(fdSplitNormalLogPdf(
        daTauObs, daLatentTau, daSigPlus, daSigMinus))
    return dLogPrior + dLogScatter + dLogMeasurement


def fdRotationLogPrior(daCoefficients, dSigmaYoung, dSigmaOld, daLatentTau):
    """Weakly-informative prior for the rotation-relation parameters."""
    dA, dB, dC, dD = daCoefficients
    if not (10.0 < dD < 60.0):
        return -np.inf
    if np.any(daLatentTau < -2.0) or np.any(daLatentTau > 1.3):
        return -np.inf
    dLogPrior = -0.5 * ((dA / 0.1) ** 2 + (dB / 5.0) ** 2 + (dC / 0.1) ** 2)
    dLogPrior += fdSigmaLogPrior(dSigmaYoung, 0.5)
    dLogPrior += fdSigmaLogPrior(dSigmaOld, 0.5)
    return dLogPrior


def fdaReconstructXuvBand(daTau, daLogLxBol):
    """Shift per-star log(LX/Lbol) onto the X-UV(5-1700A) band via published fits."""
    daShift = fdaHinge(DA_XUV_PUBLISHED, daTau) - fdaHinge(DA_LXBOL_PUBLISHED, daTau)
    return daLogLxBol + daShift


def ftLoadXuvActivityData():
    """Return (log_age, log_age_err, y_xuv, y_err) for the mid-late M sample."""
    listRows = list(csv.DictReader(
        open(os.path.join(S_DIRECTORY, "xrayActivityData.csv"))))
    daTau = np.array([float(r["log_age_gyr"]) for r in listRows])
    daTauErr = np.array([float(r["log_age_err"]) for r in listRows])
    daLogLxBol = np.array([float(r["log_lx_lbol"]) for r in listRows])
    daYErr = np.array([float(r["log_lx_lbol_err"]) for r in listRows])
    daYXuv = fdaReconstructXuvBand(daTau, daLogLxBol)
    return daTau, daTauErr, daYXuv, daYErr


def fdXuvLogPosterior(daParameters, daTau, daTauErr, daY, daYErr):
    """Log posterior for the X-UV relation with analytic error marginalization."""
    daCoefficients = daParameters[:4]
    dSigmaYoung, dSigmaOld = np.exp(daParameters[4]), np.exp(daParameters[5])
    dLogPrior = fdXuvLogPrior(daCoefficients, dSigmaYoung, dSigmaOld)
    if not np.isfinite(dLogPrior):
        return -np.inf
    daModelY = fdaHinge(daCoefficients, daTau)
    daSlope = fdaLocalSlope(daCoefficients, daTau)
    daSigmaSegment = fdSegmentSigma(dSigmaYoung, dSigmaOld, daTau,
                                    daCoefficients[3])
    daVariance = (daYErr ** 2 + (daSlope * daTauErr) ** 2 + daSigmaSegment ** 2)
    dLogLikelihood = np.sum(-0.5 * ((daY - daModelY) ** 2 / daVariance)
                            - 0.5 * np.log(daVariance))
    return dLogPrior + dLogLikelihood


def fdXuvLogPrior(daCoefficients, dSigmaYoung, dSigmaOld):
    """Weakly-informative prior for the X-UV-relation parameters."""
    dA, dB, dC, dD = daCoefficients
    if not (-0.5 < dD < 1.0):
        return -np.inf
    dLogPrior = -0.5 * ((dA / 1.0) ** 2 + ((dB + 3.0) / 2.0) ** 2
                        + (dC / 3.0) ** 2)
    dLogPrior += fdSigmaLogPrior(dSigmaYoung, 0.7)
    dLogPrior += fdSigmaLogPrior(dSigmaOld, 0.7)
    return dLogPrior


def fdaRunSampler(fdLogPosterior, daInitialGuess, tArgs, iWalkers, iSteps):
    """Run an emcee ensemble sampler and return the flattened post-burn chain."""
    np.random.seed(I_SEED)
    iDimensions = len(daInitialGuess)
    daStart = daInitialGuess + 1e-4 * np.random.randn(iWalkers, iDimensions)
    sampler = emcee.EnsembleSampler(iWalkers, iDimensions, fdLogPosterior,
                                    args=tArgs)
    sampler.run_mcmc(daStart, iSteps, progress=False)
    iBurn = iSteps // 3
    return sampler.get_chain(discard=iBurn, flat=True)


def fdaFitRotationRelation():
    """Return the flattened rotation-relation posterior chain and its data."""
    daProt, daAge, daPlus, daMinus = ftLoadRotationBenchmarks()
    daTauObs, daSigPlus, daSigMinus = ftBuildRotationLogAgeErrors(
        daAge, daPlus, daMinus)
    daInitial = np.concatenate([DA_ROT_PUBLISHED,
                                np.log([0.2, 0.1]), daTauObs])
    daChain = fdaRunSampler(
        fdRotationLogPosterior, daInitial,
        (daProt, daTauObs, daSigPlus, daSigMinus), 64, 6000)
    return daChain, daProt, daTauObs


def fdaFitXuvRelation():
    """Return the flattened X-UV-relation posterior chain and its data."""
    daTau, daTauErr, daY, daYErr = ftLoadXuvActivityData()
    daInitial = np.concatenate([DA_XUV_PUBLISHED, np.log([0.3, 0.3])])
    daChain = fdaRunSampler(
        fdXuvLogPosterior, daInitial, (daTau, daTauErr, daY, daYErr), 48, 6000)
    return daChain, daTau, daY


def fdictSummarizeCoefficients(daChain, daPublished, sLabel):
    """Return a dict comparing posterior coefficients to the published values."""
    saNames = ["a", "b", "c", "d"]
    dictSummary = {"relation": sLabel}
    for iIndex, sName in enumerate(saNames):
        daColumn = daChain[:, iIndex]
        dictSummary[sName] = {
            "posterior_mean": float(np.mean(daColumn)),
            "posterior_std": float(np.std(daColumn)),
            "published": float(daPublished[iIndex]),
        }
    dictSummary["sigma_int_young"] = fdictScatterStats(np.exp(daChain[:, 4]))
    dictSummary["sigma_int_old"] = fdictScatterStats(np.exp(daChain[:, 5]))
    return dictSummary


def fdictScatterStats(daSigma):
    """Return mean and 68% interval for a scatter posterior."""
    return {
        "mean": float(np.mean(daSigma)),
        "ci68": [float(np.percentile(daSigma, 16)),
                 float(np.percentile(daSigma, 84))],
    }


def fnSaveCoefficientCovariance(daChain, sPath):
    """Save the 4x4 coefficient covariance matrix for the relation."""
    daCovariance = np.cov(daChain[:, :4], rowvar=False)
    np.save(sPath, daCovariance)


def fnWriteJson(dictData, sPath):
    """Serialize a dictionary to a formatted JSON file."""
    with open(sPath, "w") as fileHandle:
        json.dump(dictData, fileHandle, indent=2)


def fnPrintComparison(dictSummary):
    """Print a posterior-vs-published comparison for one relation."""
    print(f"\n=== {dictSummary['relation']} relation ===")
    for sName in ["a", "b", "c", "d"]:
        dEntry = dictSummary[sName]
        print(f"  {sName}: refit {dEntry['posterior_mean']:+.4f} "
              f"+/- {dEntry['posterior_std']:.4f}  "
              f"(published {dEntry['published']:+.4f})")
    print(f"  sigma_int young: {dictSummary['sigma_int_young']['mean']:.4f}")
    print(f"  sigma_int old:   {dictSummary['sigma_int_old']['mean']:.4f}")


def main():
    """Refit both Engle relations and persist posterior products."""
    np.random.seed(I_SEED)
    daRotChain, _, _ = fdaFitRotationRelation()
    daXuvChain, _, _ = fdaFitXuvRelation()

    np.save(os.path.join(S_DIRECTORY, "rotationCoefficientSamples.npy"),
            daRotChain[:, :6])
    np.save(os.path.join(S_DIRECTORY, "xuvCoefficientSamples.npy"),
            daXuvChain[:, :6])
    fnSaveCoefficientCovariance(
        daRotChain, os.path.join(S_DIRECTORY, "rotationCovariance.npy"))
    fnSaveCoefficientCovariance(
        daXuvChain, os.path.join(S_DIRECTORY, "xuvCovariance.npy"))

    dictRot = fdictSummarizeCoefficients(daRotChain, DA_ROT_PUBLISHED, "rotation")
    dictXuv = fdictSummarizeCoefficients(daXuvChain, DA_XUV_PUBLISHED, "xuv")
    fnWriteJson({"rotation": dictRot, "xuv": dictXuv},
                os.path.join(S_DIRECTORY, "refitSummary.json"))
    fnPrintComparison(dictRot)
    fnPrintComparison(dictXuv)
    print("\nRefit complete. Posterior products written.")


if __name__ == "__main__":
    main()
