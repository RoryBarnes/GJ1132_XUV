"""Re-derive the Engle (2024, ApJ 960, 62) L_X -> L_XUV(5-1700 A) band conversion.

Engle (2024) Section "X-UV irradiance" states only that spectral energy
distributions (SEDs) constructed by the MUSCLES and Mega-MUSCLES surveys were
obtained from MAST (https://archive.stsci.edu/prepds/muscles/) and "used to
determine integrated fluxes over the 5 -- 1700 A range", yielding

    log10(L_XUV(5-1700A) / L_bol) = 0.5728 [0.0589] * log10(L_X / L_bol)
                                    - 1.0509 [0.2921]

He published neither the star list, the L_X band, the L_bol source, the
fitting method, the slope-intercept covariance, nor the intrinsic scatter.
This script reconstructs the fit and recovers the unpublished quantities.

Documented judgment calls (the paper text is silent on all of these):

1. Star list: all 24 targets with MUSCLES-family SEDs available before the
   paper's submission (2023 October): the 11 MUSCLES Treasury stars (7 M
   dwarfs and the 4 K dwarfs eps Eri, HD 40307, HD 85512, HD 97658), plus
   Proxima Centauri (GJ 551, distributed as a v22 MUSCLES product), plus
   the 12 Mega-MUSCLES M dwarfs. The full 24-star sample gives
   slope/intercept of 0.597/-0.890 (0.42/0.55 of Engle's quoted sigmas
   away), which suggests Engle fit the complete MUSCLES + Mega-MUSCLES
   sample, while the 20 M dwarfs alone give 0.652/-0.650 (1.35/1.37 sigma
   away). By project decision (2026-11-07) the M-dwarf-only fit is ADOPTED
   and written as "adopted_fit", the key every downstream step reads,
   because the relation is applied to an M dwarf (GJ 1132) and the K dwarfs
   flatten the slope at low activity; the 24-star fit is kept as an
   alternate. The v24 "MUSCLES Extension" (mostly FGK transiting-planet
   hosts, released days before submission) is excluded.
2. SED product: the "adapt-const-res-sed" panchromatic SEDs (1 A bins,
   adaptively downsampled to avoid negative fluxes), which are the
   recommended integration products.
3. SED versions: v22 for MUSCLES stars and v25 for Mega-MUSCLES stars.
   Engle would have used the v23 Mega-MUSCLES release, but MAST has
   superseded v23 with v25 and no longer distributes it; this is a known,
   unavoidable source of star-level differences.
4. L_X band: the ROSAT 0.1-2.4 keV band (5.17-124.0 A), because Engle's
   log10(L_X/L_bol) age relations are anchored to ROSAT-era soft X-ray
   fluxes. A 5-100 A alternative (the band above which the MUSCLES SEDs
   switch from APEC coronal models to Lya-scaled EUV estimates) is also
   computed on the adopted M-dwarf sample and reported as a sensitivity
   check. (On the 24-star sample it gives 0.517/-1.220, so there the two
   candidate bands bracket Engle's published coefficients from opposite
   sides.)
5. L_bol: taken from the SED products themselves via the BOLOFLUX column
   (flux density normalized by the bolometric flux, which includes a
   blackbody extension beyond 5.5 microns). Band integrals of BOLOFLUX are
   therefore L_band/L_bol directly, and distance cancels exactly.
6. Fitting: Engle quotes symmetric parameter errors with no weighting
   scheme, consistent with unweighted least squares. We report (a) ordinary
   least squares with the full 2x2 parameter covariance, and (b) a
   maximum-likelihood / Bayesian fit whose per-star variance is a single
   constant intrinsic-scatter term sigma^2 (the Kelly 2007, ApJ 665, 1489
   likelihood in the limit of negligible measurement errors; see also Hogg,
   Bovy & Lang 2010, arXiv:1008.4686, Section 7). With a Jeffreys prior
   1/sigma^2 the marginal posterior of sigma^2 is
   InverseGamma((N-2)/2, ResidualSumOfSquares/2), from which the scatter
   credible interval is drawn.
7. L 980-5: its SED X-ray segment is based on a Chandra non-detection upper
   limit (MUSCLES README); it is retained but flagged in the per-star
   provenance, and the M-dwarf fit is repeated without it as a robustness
   check.

Outputs (written next to this script):
    seds/                  cached MUSCLES SED FITS files
    perStarBandFluxes.json per-star band ratios and provenance
    conversionFit.json     fit coefficients, covariance, scatter, comparison
"""

import json
import os
import sys
import urllib.request

import numpy
from astropy.io import fits
from scipy import stats

sDirectoryBase = os.path.dirname(os.path.abspath(__file__))
sDirectorySeds = os.path.join(sDirectoryBase, "seds")
sUrlArchiveBase = "https://archive.stsci.edu/missions/hlsp/muscles"

fKiloElectronVoltAngstrom = 12.398419843320026
fBandXrayRosatLow = fKiloElectronVoltAngstrom / 2.4
fBandXrayRosatHigh = fKiloElectronVoltAngstrom / 0.1
fBandXrayApecLow = 5.0
fBandXrayApecHigh = 100.0
fBandXuvLow = 5.0
fBandXuvHigh = 1700.0

fSlopeEngle = 0.5728
fSlopeErrorEngle = 0.0589
fInterceptEngle = -1.0509
fInterceptErrorEngle = 0.2921

listTargets = [
    {"sStar": "eps Eri", "sSlug": "v-eps-eri", "sVersion": "v22", "sSurvey": "MUSCLES", "sClassGroup": "K"},
    {"sStar": "HD 40307", "sSlug": "hd40307", "sVersion": "v22", "sSurvey": "MUSCLES", "sClassGroup": "K"},
    {"sStar": "HD 85512", "sSlug": "hd85512", "sVersion": "v22", "sSurvey": "MUSCLES", "sClassGroup": "K"},
    {"sStar": "HD 97658", "sSlug": "hd97658", "sVersion": "v22", "sSurvey": "MUSCLES", "sClassGroup": "K"},
    {"sStar": "GJ 176", "sSlug": "gj176", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 436", "sSlug": "gj436", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 551", "sSlug": "gj551", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 581", "sSlug": "gj581", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 667C", "sSlug": "gj667c", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 832", "sSlug": "gj832", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 876", "sSlug": "gj876", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 1214", "sSlug": "gj1214", "sVersion": "v22", "sSurvey": "MUSCLES"},
    {"sStar": "GJ 15A", "sSlug": "gj15a", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 163", "sSlug": "gj163", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 649", "sSlug": "gj649", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 674", "sSlug": "gj674", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 676A", "sSlug": "gj676a", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 699", "sSlug": "gj699", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 729", "sSlug": "gj729", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 849", "sSlug": "gj849", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "GJ 1132", "sSlug": "gj1132", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "L 980-5", "sSlug": "l-980-5", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "LHS 2686", "sSlug": "lhs-2686", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
    {"sStar": "TRAPPIST-1", "sSlug": "trappist-1", "sVersion": "v25", "sSurvey": "Mega-MUSCLES"},
]


def fsBuildSedFileName(dictTarget):
    return "hlsp_muscles_multi_multi_{}_broadband_{}_adapt-const-res-sed.fits".format(
        dictTarget["sSlug"], dictTarget["sVersion"]
    )


def fsBuildSedUrl(dictTarget):
    return "{}/{}/{}/{}".format(
        sUrlArchiveBase,
        dictTarget["sVersion"],
        dictTarget["sSlug"],
        fsBuildSedFileName(dictTarget),
    )


def fsDownloadSedIfMissing(dictTarget):
    sPathLocal = os.path.join(sDirectorySeds, fsBuildSedFileName(dictTarget))
    if os.path.exists(sPathLocal) and os.path.getsize(sPathLocal) > 100000:
        return sPathLocal
    sUrlRemote = fsBuildSedUrl(dictTarget)
    print("Downloading {} ...".format(sUrlRemote))
    try:
        urllib.request.urlretrieve(sUrlRemote, sPathLocal)
    except Exception as oError:
        raise RuntimeError(
            "Failed to download {}: {}".format(sUrlRemote, oError)
        ) from oError
    return sPathLocal


def ffIntegrateBandRatio(daEdgeLow, daEdgeHigh, daBolometricRatio, fBandLow, fBandHigh):
    """Integrate the BOLOFLUX density over a band with fractional edge bins."""
    daOverlap = numpy.minimum(daEdgeHigh, fBandHigh) - numpy.maximum(daEdgeLow, fBandLow)
    daOverlap = numpy.clip(daOverlap, 0.0, None)
    baFinite = numpy.isfinite(daBolometricRatio)
    return float(numpy.sum(daOverlap[baFinite] * daBolometricRatio[baFinite]))


def fdictComputeStarBandRatios(dictTarget, sPathFits):
    with fits.open(sPathFits) as listHduList:
        daEdgeLow = numpy.asarray(listHduList[1].data["WAVELENGTH0"], dtype=float)
        daEdgeHigh = numpy.asarray(listHduList[1].data["WAVELENGTH1"], dtype=float)
        daBolometricRatio = numpy.asarray(listHduList[1].data["BOLOFLUX"], dtype=float)
        fBolometricFlux = float(listHduList[0].header["BOLOFLUX"])
    fRatioXuv = ffIntegrateBandRatio(daEdgeLow, daEdgeHigh, daBolometricRatio, fBandXuvLow, fBandXuvHigh)
    fRatioXrayRosat = ffIntegrateBandRatio(daEdgeLow, daEdgeHigh, daBolometricRatio, fBandXrayRosatLow, fBandXrayRosatHigh)
    fRatioXrayApec = ffIntegrateBandRatio(daEdgeLow, daEdgeHigh, daBolometricRatio, fBandXrayApecLow, fBandXrayApecHigh)
    return {
        "star": dictTarget["sStar"],
        "survey": dictTarget["sSurvey"],
        "spectral_class_group": dictTarget.get("sClassGroup", "M"),
        "sed_file": fsBuildSedFileName(dictTarget),
        "sed_url": fsBuildSedUrl(dictTarget),
        "bolometric_flux_erg_s_cm2": fBolometricFlux,
        "log_lx_lbol": float(numpy.log10(fRatioXrayRosat)),
        "log_lx_lbol_band_angstrom": [fBandXrayRosatLow, fBandXrayRosatHigh],
        "log_lx_lbol_alternate_5_100": float(numpy.log10(fRatioXrayApec)),
        "log_lxuv_lbol": float(numpy.log10(fRatioXuv)),
        "provenance_note": fsDescribeProvenance(dictTarget),
    }


def fsDescribeProvenance(dictTarget):
    sNote = (
        "Band ratios integrated from the BOLOFLUX column of the MAST "
        "{} adapt-const-res SED; L_bol from the SED BOLOFLUX header keyword."
    ).format(dictTarget["sVersion"])
    if dictTarget["sSlug"] == "l-980-5":
        sNote += " CAUTION: X-ray segment based on a Chandra non-detection upper limit."
    if dictTarget["sSlug"] == "gj551":
        sNote += " Proxima Cen v22 add-on; see MUSCLES reduction notes on its T_eff/F_bol."
    if dictTarget["sSlug"] == "hd97658":
        sNote += " CAUTION: X-ray segment scaled from HD 85512 (no direct X-ray data)."
    return sNote


def fdictFitOrdinaryLeastSquares(daAbscissa, daOrdinate):
    """Unweighted straight-line fit with the full 2x2 parameter covariance."""
    iCount = len(daAbscissa)
    daDesign = numpy.column_stack([daAbscissa, numpy.ones(iCount)])
    daCoefficients, _, _, _ = numpy.linalg.lstsq(daDesign, daOrdinate, rcond=None)
    daResiduals = daOrdinate - daDesign @ daCoefficients
    fResidualSumOfSquares = float(numpy.sum(daResiduals**2))
    fVarianceEstimate = fResidualSumOfSquares / (iCount - 2)
    daCovariance = fVarianceEstimate * numpy.linalg.inv(daDesign.T @ daDesign)
    return {
        "fSlope": float(daCoefficients[0]),
        "fIntercept": float(daCoefficients[1]),
        "daCovariance": daCovariance,
        "fResidualSumOfSquares": fResidualSumOfSquares,
        "iCount": iCount,
    }


def fdictEstimateIntrinsicScatter(fResidualSumOfSquares, iCount):
    """Posterior of the intrinsic scatter sigma under a Jeffreys prior.

    With negligible measurement errors, the Kelly (2007) regression
    likelihood reduces to y ~ Normal(slope*x + intercept, sigma^2); the
    marginal posterior is sigma^2 ~ InverseGamma((N-2)/2, RSS/2).
    """
    oPosterior = stats.invgamma(a=(iCount - 2) / 2.0, scale=fResidualSumOfSquares / 2.0)
    fScatterMaximumLikelihood = numpy.sqrt(fResidualSumOfSquares / iCount)
    return {
        "fScatterMaximumLikelihood": float(fScatterMaximumLikelihood),
        "fScatterPosteriorMedian": float(numpy.sqrt(oPosterior.median())),
        "fScatterCredible16": float(numpy.sqrt(oPosterior.ppf(0.16))),
        "fScatterCredible84": float(numpy.sqrt(oPosterior.ppf(0.84))),
        "fScatterCredible2p5": float(numpy.sqrt(oPosterior.ppf(0.025))),
        "fScatterCredible97p5": float(numpy.sqrt(oPosterior.ppf(0.975))),
    }


def fdictDescribeCovariance(daCovariance):
    fCorrelation = daCovariance[0, 1] / numpy.sqrt(daCovariance[0, 0] * daCovariance[1, 1])
    return {
        "slope_error": float(numpy.sqrt(daCovariance[0, 0])),
        "intercept_error": float(numpy.sqrt(daCovariance[1, 1])),
        "covariance_slope_intercept": [[float(daCovariance[0, 0]), float(daCovariance[0, 1])],
                                       [float(daCovariance[1, 0]), float(daCovariance[1, 1])]],
        "slope_intercept_correlation": float(fCorrelation),
    }


def fdictSummarizeLineFit(daAbscissa, daOrdinate):
    dictLeastSquares = fdictFitOrdinaryLeastSquares(daAbscissa, daOrdinate)
    dictScatter = fdictEstimateIntrinsicScatter(
        dictLeastSquares["fResidualSumOfSquares"], dictLeastSquares["iCount"]
    )
    fMeanAbscissa = float(numpy.mean(daAbscissa))
    dictSummary = {"slope": dictLeastSquares["fSlope"], "intercept": dictLeastSquares["fIntercept"]}
    dictSummary.update(fdictDescribeCovariance(dictLeastSquares["daCovariance"]))
    dictSummary.update({
        "intrinsic_scatter_dex": dictScatter,
        "n_stars": dictLeastSquares["iCount"],
        "pivot": 0.0,
        "pivot_note": "Fit performed about pivot x=0 so the intercept matches Engle's convention.",
        "mean_abscissa": fMeanAbscissa,
        "intercept_at_mean_abscissa": float(
            dictLeastSquares["fSlope"] * fMeanAbscissa + dictLeastSquares["fIntercept"]
        ),
    })
    return dictSummary


def fdictCompareToEngle(dictFit):
    fSlopeDifference = dictFit["slope"] - fSlopeEngle
    fInterceptDifference = dictFit["intercept"] - fInterceptEngle
    return {
        "engle_slope": fSlopeEngle,
        "engle_slope_error": fSlopeErrorEngle,
        "engle_intercept": fInterceptEngle,
        "engle_intercept_error": fInterceptErrorEngle,
        "slope_difference": fSlopeDifference,
        "slope_difference_in_engle_sigma": fSlopeDifference / fSlopeErrorEngle,
        "intercept_difference": fInterceptDifference,
        "intercept_difference_in_engle_sigma": fInterceptDifference / fInterceptErrorEngle,
    }


def flistGatherPerStarRatios():
    os.makedirs(sDirectorySeds, exist_ok=True)
    listPerStar = []
    for dictTarget in listTargets:
        sPathFits = fsDownloadSedIfMissing(dictTarget)
        listPerStar.append(fdictComputeStarBandRatios(dictTarget, sPathFits))
    return listPerStar


def fnWriteJson(sFileName, dictPayload):
    sPathOutput = os.path.join(sDirectoryBase, sFileName)
    with open(sPathOutput, "w") as fileOutput:
        json.dump(dictPayload, fileOutput, indent=2)
    print("Wrote {}".format(sPathOutput))


def fdaExtractColumn(listPerStar, sKey):
    return numpy.array([dictStar[sKey] for dictStar in listPerStar])


def fdictBuildConversionReport(listPerStar):
    """Fit the adopted M-dwarf sample, its sensitivity checks, and the all-star alternate."""
    daAbscissa = fdaExtractColumn(listPerStar, "log_lx_lbol")
    daOrdinate = fdaExtractColumn(listPerStar, "log_lxuv_lbol")
    daAbscissaAlternate = fdaExtractColumn(listPerStar, "log_lx_lbol_alternate_5_100")
    baOnlyMDwarfs = fdaExtractColumn(listPerStar, "spectral_class_group") == "M"
    baMDwarfDetections = baOnlyMDwarfs & (fdaExtractColumn(listPerStar, "star") != "L 980-5")
    dictFitAdopted = fdictSummarizeLineFit(daAbscissa[baOnlyMDwarfs], daOrdinate[baOnlyMDwarfs])
    dictFitAllTargets = fdictSummarizeLineFit(daAbscissa, daOrdinate)
    return {
        "relation": "log10(L_XUV(5-1700A)/L_bol) = slope * log10(L_X/L_bol) + intercept",
        "adopted_sample": "M dwarfs only, ROSAT 0.1-2.4 keV band",
        "adopted_fit": dictFitAdopted,
        "engle_comparison": fdictCompareToEngle(dictFitAdopted),
        "alternate_fit_all_targets_rosat_band": dictFitAllTargets,
        "engle_comparison_all_targets": fdictCompareToEngle(dictFitAllTargets),
        "alternate_fit_m_dwarfs_lx_5_100_angstrom": fdictSummarizeLineFit(
            daAbscissaAlternate[baOnlyMDwarfs], daOrdinate[baOnlyMDwarfs]),
        "robustness_fit_m_dwarfs_without_l980_5_upper_limit": fdictSummarizeLineFit(
            daAbscissa[baMDwarfDetections], daOrdinate[baMDwarfDetections]),
        "method_notes": fsSummarizeMethodNotes(),
    }


def fsSummarizeMethodNotes():
    return (
        "Adopted fit: unweighted ordinary least squares on the 20 M dwarfs "
        "(MUSCLES v22 including Proxima Cen, Mega-MUSCLES v25); the 24-target fit "
        "that adds the 4 MUSCLES K dwarfs is kept as an alternate. Parameter "
        "covariance uses sigma^2 = RSS/(N-2). Intrinsic scatter follows the Kelly "
        "(2007, ApJ 665, 1489) likelihood with negligible measurement errors: "
        "per-star variance is a single constant sigma^2, whose posterior under a "
        "Jeffreys prior is InverseGamma((N-2)/2, RSS/2). L_X band is ROSAT "
        "0.1-2.4 keV (5.17-124 A); a 5-100 A alternative and a fit without the "
        "L 980-5 X-ray upper limit are computed on the M-dwarf sample as "
        "sensitivity checks. All ratios come from the SED BOLOFLUX column, so "
        "distances cancel. Engle used the superseded Mega-MUSCLES v23 release, "
        "which MAST no longer distributes; see module docstring for all "
        "documented judgment calls."
    )


def main():
    listPerStar = flistGatherPerStarRatios()
    fnWriteJson("perStarBandFluxes.json", {"stars": listPerStar})
    dictReport = fdictBuildConversionReport(listPerStar)
    fnWriteJson("conversionFit.json", dictReport)
    dictPrimary = dictReport["adopted_fit"]
    print("slope    = {:.4f} +/- {:.4f}".format(dictPrimary["slope"], dictPrimary["slope_error"]))
    print("intercept = {:.4f} +/- {:.4f}".format(dictPrimary["intercept"], dictPrimary["intercept_error"]))
    print("corr(slope, intercept) = {:.4f}".format(dictPrimary["slope_intercept_correlation"]))
    print("intrinsic scatter (median [68%]) = {:.4f} [{:.4f}, {:.4f}] dex".format(
        dictPrimary["intrinsic_scatter_dex"]["fScatterPosteriorMedian"],
        dictPrimary["intrinsic_scatter_dex"]["fScatterCredible16"],
        dictPrimary["intrinsic_scatter_dex"]["fScatterCredible84"]))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as oError:
        print("ERROR: {}".format(oError), file=sys.stderr)
        sys.exit(1)
