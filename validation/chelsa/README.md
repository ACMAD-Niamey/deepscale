# CHELSA precipitation reproduction gate

This directory records the scientific reproduction gate used for the public
DeepScale method. The project decision is to treat the published equations as
authoritative; archived executable behavior is diagnostic where it conflicts
with those equations.

## Canonical target

The target is the precipitation branch of CHELSA-W5E5 v1.0 / CHELSA V2,
described by Karger et al. (2023), with the detailed precipitation
parameterization inherited from Karger et al. (2021).  It is not the removed
`terrain-disagg` random-forest prototype.

Primary sources:

- Karger et al. (2021), *Global daily 1 km land surface precipitation based on
  cloud cover-informed downscaling*, DOI `10.1038/s41597-021-01084-6`.
- Karger et al. (2023), *CHELSA-W5E5: daily 1 km meteorological forcing data
  for climate impact studies*, DOI `10.5194/essd-15-2445-2023`.
- Archived CHELSA-W5E5 v1.0 source, DOI `10.5281/zenodo.8010301`, commit
  `01bec3401f961bf4c4ee70c3df40fac1b4a5ea5c`.
- SAGA GIS `ta_morphometry/15`, Wind Effect (Windward / Leeward Index).

The official CHELSA source is GPL-3.0.  DeepScale is MIT licensed, so the
eventual implementation must remain an independent implementation of the
published equations and externally observable reference behavior.  Do not
copy SAGA or CHELSA implementation code into `src/deepscale`.

## Equation-to-operation map

1. Interpolate daily mean zonal and meridional 10 m wind to the terrain grid
   and convert it to direction.  CHELSA-W5E5 uses multilevel B-splines and a
   World Mercator 3 km working grid.
2. Calculate the SAGA windward/leeward index `H`.  The archived W5E5 driver
   calls SAGA with a 300 km search distance, acceleration 1.5, variable wind
   direction, and the non-legacy algorithm.
3. Correct `H` for distance to planetary-boundary-layer elevation.  In the
   2021 paper, `PBL_z = PBL + z_coarse + 500 m` and the correction scale is
   9000 m.
4. Apply the high-elevation exposition correction `E`; the paper defines the
   first precipitation intensity approximation as `p_I = E * H_B`.
5. Normalize `p_I` within every coarse parent cell and multiply it by coarse
   precipitation.  Equation 24 of Karger et al. (2023) requires the mean of
   the fine cells in a complete parent cell to equal its coarse precipitation
   flux.
6. CHELSA-W5E5 omits the satellite cloud-frequency refinement used by
   CHELSA-EarthEnv.

## Source ambiguity that must be resolved

The archived `obs/CHELSA_W5E5.py` computes a PBL-corrected field as
`windef / (1 - correction/9000)` and then immediately assigns
`expocor * windef` to the same variable.  Taken literally, this discards the
PBL correction.  That conflicts with both papers, which describe the
precipitation intensity as the exposition correction times the PBL-corrected
wind effect.

Therefore there are two separately named reproduction targets:

- `paper`: `p_I = E * H_B`, including the 500 m PBL offset from 2021.
- `archived-code`: reproduce the archived executable behavior, including the
  apparent overwrite if an upstream run confirms it.

DeepScale implements the `paper` target and records
`chelsa_variant="paper-defined Karger 2021/2023"` in its output metadata. The
archived-code discrepancy is documented rather than silently reproduced.

## Existing evidence

The synthetic 9x9 ridge fixture in `saga_wind_effect.csv` was produced by
SAGA's Wind Effect tool on a 5 km projected grid.  The independent prototype
has mean absolute error about 0.0044 against it; the largest differences are
edge cells.  This validates the central wind-effect calculation only.  It is
not end-to-end CHELSA parity.

## Admission checklist

- [ ] Pin and archive the 2021 paper alongside its checksum.
- [x] Record exact SAGA and CHELSA versions used for the component reference run.
- [ ] Produce one small upstream end-to-end fixture containing coarse
      precipitation, DEM, wind, PBL, coarse orography, exposition correction,
      intermediate `H`, intermediate `H_B`, and final precipitation.
- [ ] Reproduce the upstream fixture within tolerances justified by its stored
      precision, including boundary cells.
- [x] Demonstrate exact parent-cell conservation for complete parent cells.
- [x] Test multiple forecast members and reject invalid precipitation/inputs.
- [x] Establish an operational no-leakage contract for forecast wind and PBL.
- [ ] Validate the mountain benefit against independent observations and a
      domain larger than the 13-cell Rwanda MAM subset.
- [x] Add `src/deepscale/methods/chelsa.py`, register `"chelsa"`, and update the
      skill, method, troubleshooting, and README documentation under the
      explicit paper-authoritative policy.
