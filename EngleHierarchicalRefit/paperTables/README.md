# Verbatim benchmark tables from the Engle "Living with a Red Dwarf" papers

Extracted 2026-07-13 from arXiv TeX sources for the Stage-1 hierarchical refit
(all M-dwarf sub-classes). Values are verbatim from the published deluxetables;
the `age_via` annotations encode fit-membership facts stated in the papers.

## Sources

- `rotationEarlyM.csv`, `rotationMidM.csv` — Engle & Guinan (2023), ApJL 954,
  L50 (arXiv:2307.01136v3), Tables 2 (M0-2) and 3 (M2.5-6.5).
- `earlyMXray.csv`, `midlateMXray.csv` — Engle (2024), ApJ 960, 62
  (arXiv:2310.04302v2), Tables 3 (M0-2) and 4 (M2.5-6.5). Columns include
  Prot, log L_X, and provenance flags absent from the older working file
  `../xrayActivityData.csv` (which matches Table 4 exactly, row for row).

## Key facts recorded during extraction (verified against the TeX)

1. Field-star ages in the X-ray tables are DERIVED from the Engle & Guinan
   (2023) rotation relations applied to the tabulated Prot. Only the cluster
   rows, GJ 191, and (by shared adopted value) GJ 699/GJ 273/GJ 581 have
   independent ages. Field-star age errors therefore share the gyro relation's
   systematic and are not independent.
2. L_X -> X-UV(5-1700 A) conversion is Engle's own MUSCLES/Mega-MUSCLES SED
   fit, NOT Sanz-Forcada:
   log(L_XUV/L_bol) = 0.5728[+/-0.0589] * log(L_X/L_bol) - 1.0509[+/-0.2921].
   Slope-intercept covariance and fit scatter are unpublished.
3. The papers' fits use numpy.piecewise + scipy least_squares; for
   single-measurement stars the X-ray "error" is a substituted average
   variability scatter (e.g. the repeated 0.36 dex log_lx_lbol_err values).
4. M67 appears in the rotation tables but is EXCLUDED from the published fits
   (its literature age is gyro-based -> circular).
5. Halo subdwarfs ARE included in the published rotation fits, assigned by
   interior structure (Kapteyn's sdM1.5 is grouped mid-late).
6. Sub-class membership is uncertain for the stars printed in bold in the
   paper (12 mid-late, 14 early); see the agent extraction reports in the
   session transcript of 2026-07-13 for the lists.
7. Paper-internal oddities: GJ 625 appears twice in the early X-ray table with
   different Prot; GJ 752 A appears in both X-ray tables with different ages;
   cluster anchor ages differ slightly between the two X-ray tables.
