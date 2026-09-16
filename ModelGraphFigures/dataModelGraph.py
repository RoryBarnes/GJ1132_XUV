#!/usr/bin/env python3
"""Assemble the node/edge/plate structure for the GJ 1132 model figure.

This script emits probabilisticGraphicalModelStructure.json, the
machine-readable description that plotProbabilisticGraphicalModel.py renders
as the generative PGM. The structure is derived by hand from the workflow
dependency graph (``.vaibify/workflows/gj1132-xuv.json``) and from the
manuscript model (``gj1132.tex``: Eqs. lxuv, englerot, englexuv, ffd,
LXUVFlare). Keeping the structure in a data step (rather than hard-coded
inside the renderer) makes the parameter categorization auditable and
version-controlled.

NOTE: the joint-refit restructuring (2026-07) replaced the Sanz-Forcada
conversion with the re-derived MUSCLES relation and added the per-star
scatter offset z; the node/edge lists below still describe the pre-refit
model and are updated together with the sweep rewiring.
"""

import json
import os

S_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
F_PGM_SCALE = 3.83


def flistPgmNodes():
    """Return the PGM node records with category and grid coordinates."""
    return _flistPgmConstants() + _flistPgmLatent() + \
        _flistPgmDeterministic() + _flistPgmObserved()


def _flistPgmConstants():
    """Constant / hyperparameter square nodes."""
    return [
        _dictPgm("h_mass", r"$m\sim\mathcal{N}$", "constant", 6.6, 11.2),
        _dictPgm("h_fsat", r"$f_{sat}\sim\mathcal{N}$", "constant", 8.4, 11.2),
        _dictPgm("h_tsat", r"$t_{sat}\sim\mathcal{U}$", "constant", 10.2, 11.2),
        _dictPgm("h_beta", r"$\beta\sim\mathcal{N}$", "constant", 12.0, 11.2),
        _dictPgm("h_flare", r"$E_{min},E_{max}$", "constant", 18.8, 6.4,
                 sNote="Osten-Wolk fractions, 5-1700A band"),
        _dictPgm("h_sanz", "Sanz-\nForcada", "constant", 4.4, 2.2,
                 sNote="X->XUV scaling coeffs"),
        _dictPgm("h_metal", "[Fe/H]", "constant", 4.8, 8.8,
                 sNote="solar Baraffe track"),
    ]


def _flistPgmLatent():
    """Stochastic / latent open-circle nodes (parameters to infer)."""
    return [
        _dictPgm("th_rot", r"$\theta_{rot}$", "stochastic", 1.4, 9.4,
                 sNote="(a,b,c,d)_rot + sigma_int"),
        _dictPgm("th_xuv", r"$\theta_{xuv}$", "stochastic", 3.9, 9.4,
                 sNote="(a,b,c,d)_xuv + sigma_int"),
        _dictPgm("tau", r"$\tau$", "stochastic", 4.8, 7.2,
                 sNote="GJ 1132 latent log age"),
        _dictPgm("m", r"$m_\star$", "stochastic", 6.6, 9.4),
        _dictPgm("fsat", r"$f_{sat}$", "stochastic", 8.4, 9.4),
        _dictPgm("tsat", r"$t_{sat}$", "stochastic", 10.2, 9.4),
        _dictPgm("beta", r"$\beta$", "stochastic", 12.0, 9.4,
                 sNote="beta_XUV"),
        _dictPgm("th_ffd", r"$\theta_{FFD}$", "stochastic", 17.0, 10.4,
                 sNote="a1,a2,a3,b1,b2,b3"),
        _dictPgm("alpha_k", r"$\alpha_k$", "stochastic", 0.9, 7.6,
                 sNote="per-star latent log-age"),
    ]


def _flistPgmDeterministic():
    """Deterministic dashed-circle nodes (functions of parents)."""
    return [
        _dictPgm("lbol", r"$L_{bol}$", "deterministic", 6.6, 7.4,
                 sNote="L_bol(t), Baraffe"),
        _dictPgm("q_emd", r"$q_{\rm E}$", "deterministic", 3.2, 5.2,
                 sNote="EMD L_XUV^quiesc/L_bol"),
        _dictPgm("q_ribas", r"$q_{\rm R}$", "deterministic", 9.2, 7.0,
                 sNote="Ribas L_XUV^quiesc/L_bol"),
        _dictPgm("ffd_ab", r"$a,b(t)$", "deterministic", 17.0, 7.4,
                 sNote="FFD slope/intercept"),
        _dictPgm("flare", r"$L^{fl}_{XUV}$", "deterministic", 17.0, 5.6,
                 sNote="flare XUV luminosity"),
        _dictPgm("total", r"$L^{tot}_{XUV}$", "deterministic", 12.6, 4.4,
                 sNote="quiescent + flare"),
        _dictPgm("semimajor", r"$a_{orb}$", "deterministic", 9.8, 4.4,
                 sNote="semi-major axis"),
        _dictPgm("xuv_t", r"$F_{XUV}(t)$", "deterministic", 11.4, 2.9),
        _dictPgm("cumflux", r"$F^{cum}_{XUV}$", "deterministic", 10.4, 1.2,
                 sNote="cumulative XUV flux (target)"),
    ]


def _flistPgmObserved():
    """Observed / clamped shaded-circle nodes (data)."""
    return [
        _dictPgm("d_rot", r"$d^{rot}_k$", "observed", 2.0, 7.6,
                 sNote="Engle rotation calibrators (P_rot, age)"),
        _dictPgm("d_xuv", r"$d^{xuv}_k$", "observed", 3.9, 7.6,
                 sNote="Engle X-ray calibrators (logLx, age)"),
        _dictPgm("d_kepler", r"$d^{K}_j$", "observed", 17.0, 8.8,
                 sNote="Kepler flare data"),
        _dictPgm("o_prot", r"$P^{obs}_{rot}$", "observed", 4.1, 5.0,
                 sNote="122.3 d, Bonfils18"),
        _dictPgm("o_lxuv", r"$q_{\rm obs}$", "observed", 6.4, 2.6,
                 sNote="log(LXUV/Lbol)=-4.26, via Sanz-Forcada"),
        _dictPgm("o_lbol", r"$L^{obs}_{bol}$", "observed", 6.4, 5.9,
                 sNote="Xue24"),
        _dictPgm("o_tess", r"$d_{\rm TESS}$", "observed", 18.8, 8.0,
                 sNote="current-rate posterior-predictive check"),
    ]


def flistPgmEdges():
    """Return directed edges (parent -> child) of the generative model."""
    return [
        _e("h_mass", "m"), _e("h_fsat", "fsat"), _e("h_tsat", "tsat"),
        _e("h_beta", "beta"), _e("h_metal", "lbol"), _e("h_flare", "flare"),
        _e("h_sanz", "o_lxuv"),
        _e("th_rot", "d_rot"), _e("alpha_k", "d_rot"), _e("th_xuv", "d_xuv"),
        _e("th_rot", "o_prot"), _e("tau", "o_prot"),
        _e("th_xuv", "q_emd"), _e("tau", "q_emd"), _e("q_emd", "o_lxuv"),
        _e("tau", "m"), _e("tau", "q_ribas"),
        _e("m", "q_ribas"), _e("fsat", "q_ribas"), _e("tsat", "q_ribas"),
        _e("beta", "q_ribas"), _e("m", "lbol"),
        _e("q_ribas", "o_lxuv"), _e("lbol", "o_lbol"),
        _e("th_ffd", "d_kepler"), _e("th_ffd", "ffd_ab"), _e("tau", "ffd_ab"),
        _e("m", "ffd_ab"), _e("ffd_ab", "flare"), _e("ffd_ab", "o_tess"),
        _e("q_ribas", "total"), _e("lbol", "total"), _e("flare", "total"),
        _e("m", "semimajor"),
        _e("total", "xuv_t"), _e("semimajor", "xuv_t"),
        _e("xuv_t", "cumflux"),
    ]


def flistPgmPlates():
    """Return plate rectangles for replicated sub-graphs."""
    return [
        {"sLabel": "rot. cal. $k$", "listNodes": ["alpha_k", "d_rot"],
         "fX": 0.2, "fY": 6.8, "fWidth": 2.5, "fHeight": 1.6,
         "sPosition": "bottom left"},
        {"sLabel": "xuv cal. $k$", "listNodes": ["d_xuv"],
         "fX": 3.45, "fY": 6.9, "fWidth": 1.1, "fHeight": 1.4,
         "sPosition": "bottom right"},
        {"sLabel": "Kepler $j$", "listNodes": ["d_kepler"],
         "fX": 16.3, "fY": 8.1, "fWidth": 1.5, "fHeight": 1.5,
         "sPosition": "top right"},
        {"sLabel": "posterior draws / vplanet runs s",
         "listNodes": ["total", "semimajor", "xuv_t", "cumflux"],
         "fX": 8.9, "fY": 0.4, "fWidth": 4.9, "fHeight": 4.9,
         "sPosition": "bottom right"},
    ]


def flistCategorizationTable():
    """Return the parameter -> category rationale table for the record."""
    return [
        _row("L_X, P_rot, L_bol, Kepler FFD, TESS flares, Engle calibrators",
             "observed", "measured data / literature values clamped in the likelihood"),
        _row("theta_rot, theta_xuv (Engle a,b,c,d + sigma_int)", "stochastic",
             "refit hierarchically with posteriors + covariance, not fixed constants"),
        _row("alpha_k per-star log-age", "stochastic",
             "latent nuisance in the rotation calibrator hierarchy (asym. errors)"),
        _row("tau (GJ 1132 age)", "stochastic",
             "inferred from P_rot; becomes the empirical age prior for Ribas"),
        _row("m_star, f_sat, t_sat, beta_XUV", "stochastic",
             "Ribas parameters inferred by alabi/MCMC under informative priors"),
        _row("theta_FFD (a1..b3)", "stochastic",
             "Davenport FFD coefficients from the emcee Kepler fit"),
        _row("L_XUV/L_bol (EMD & Ribas), a(t)/b(t), L_XUV_flare, L_XUV_tot, "
             "F_XUV(t), a_orb, F_XUV_cum", "deterministic",
             "algebraic / vplanet functions of the parameters above"),
        _row("prior hyper-params, E_min/E_max, Osten-Wolk fractions, bandpass, "
             "Sanz-Forcada coeffs, [Fe/H]", "constant",
             "fixed inputs; not inferred"),
    ]


def _dictPgm(sId, sLabel, sCategory, fX, fY, sNote=""):
    """Build a PGM node record."""
    return {"sId": sId, "sLabel": sLabel, "sCategory": sCategory,
            "fX": fX, "fY": fY, "sNote": sNote}


def _e(sSrc, sDst):
    """Build a PGM directed edge record."""
    return {"sSrc": sSrc, "sDst": sDst}


def _row(sParameter, sCategory, sRationale):
    """Build a categorization-table row."""
    return {"sParameter": sParameter, "sCategory": sCategory,
            "sRationale": sRationale}


def fnSaveJson(dictData, sFilename):
    """Write a structure dictionary to disk as indented JSON."""
    sPath = os.path.join(S_DIRECTORY, sFilename)
    with open(sPath, "w") as fileHandle:
        json.dump(dictData, fileHandle, indent=2)
    print(f"Saved {sFilename} ({len(json.dumps(dictData))} bytes)")


def _fnScalePgmLayout(dictPgm, fScale):
    """Scale PGM node and plate coordinates for generous, overlap-free spacing."""
    for dictNode in dictPgm["listNodes"]:
        dictNode["fX"] *= fScale
        dictNode["fY"] *= fScale
    for dictPlate in dictPgm["listPlates"]:
        for sKey in ("fX", "fY", "fWidth", "fHeight"):
            dictPlate[sKey] *= fScale


def fnBuildAndSave():
    """Assemble the PGM structure and write it to the step directory."""
    dictPgm = {"listNodes": flistPgmNodes(), "listEdges": flistPgmEdges(),
               "listPlates": flistPgmPlates(),
               "listCategorization": flistCategorizationTable()}
    _fnScalePgmLayout(dictPgm, F_PGM_SCALE)
    fnSaveJson(dictPgm, "probabilisticGraphicalModelStructure.json")


if __name__ == "__main__":
    fnBuildAndSave()
