"""Calibration with calibrate(): eReg, logistic/WVG, and smoothed_regression.

calibrate() is Model Output Statistics — it does NOT change resolution and
does NOT regrid. Put every gridded predictor on the obs grid first:
    hcst = hcst.interp(lat=obs.lat, lon=obs.lon)
"""

import deepscale as ds

# =========================================================================
# 1. Ensemble regression (eReg) — gridded multi-model calibration
# =========================================================================
# Per grid cell: OLS of obs on ensemble-mean hindcast -> Gaussian terciles
# with leverage-inflated prediction-error variance (Wilks 2006 eq 6.22).
# Predictor form: {model: (hindcast, forecast_or_None)}; models are
# calibrated independently, averaged, and renormalized to sum to 1.

models = {
    "ECMWF": (ecmwf_hcst, ecmwf_fcst),   # each hcst (year, member, lat, lon), on obs grid
    "CFSv2": (cfsv2_hcst, None),          # None -> forecast_year picks the year
}
probs = ds.calibrate(
    models, obs,
    method="ereg",
    forecast_year=2024,
    clip_negative=True,        # precip: clamp negative regressed values
    threshold_source="obs",    # tercile boundaries from obs (or "fitted")
    native_years=False,        # True: per-model obs-overlap calibration windows
)
assert probs.sizes["tercile"] == 3   # (tercile, lat, lon), sums to 1 per valid cell

# =========================================================================
# 2. Logistic calibration on a teleconnection index (scalar predictor)
# =========================================================================
# Named indices: "wvg" (Western-V Gradient, 3-box), "wvg2", "nino34", "nino4"
wvg = ds.Index.named("wvg")

# Or a custom index from bbox regions ([lat_s, lat_n, lon_w, lon_e], 0-360 lon ok):
wvg_custom = ds.Index.custom(
    name="wvg",
    regions={
        "nino34": [-5, 5, 190, 240],
        "wnp": [20, 35, 160, 210],
        "wep": [-15, 20, 120, 160],
        "wsp": [-30, -15, 155, 210],
    },
    combine=lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3,
)

# Gridded SST in, LogitConfig reduces it through the index automatically.
# Pass hindcast SST as the climatology when reducing forecasts so both
# share the same standardization scale (Index.reduce(sst, climatology=...)).
probs = ds.calibrate(
    predictor_hindcast=sst_hindcast,     # (year, member, lat, lon) SST
    obs=obs,                             # (year, lat, lon) predictand
    predictor_forecast=sst_forecast,     # (member, lat, lon) SST
    method=ds.LogitConfig(
        index=wvg_custom,
        model="independent_binomial",    # or "multinomial"
        predictor_level="model_mean",
        detrend=False,
        significance=None,               # set e.g. 0.1 -> statsmodels backend + mask
        regularization=None,
    ),
)

# Already-reduced scalar index series work too:
probs = ds.calibrate(index_series, obs, method="logit", forecast=1.7, min_years=10)

ds.plot_terciles(probs, title="WVG-calibrated MAM precip")

# =========================================================================
# 3. Smoothed-coefficient regression (Kharin et al. 2017) — season-aware
# =========================================================================
# Rescales the ensemble-mean anomaly with a per-cell regression coefficient
# smoothed ACROSS the seasonal cycle. Inputs carry a `season` dim this method
# owns: hindcast (season, year, member, lat, lon), obs (season, year, lat, lon),
# same grid. The fit always comes from the hindcast (CV is caller's); the target
# is EITHER forecast_year= (a year present in the hindcast, retro-forecast) OR
# forecast= (an out-of-sample ensemble, real-time) -- mutually exclusive.
#
# temporal_sigma: None (per-season) | float (cyclic Gaussian smoothing) |
#                 "constant" (one year-round coefficient).

# Deterministic: rescaled ensemble-mean anomaly (season, lat, lon). Score with `msss`.
adjusted = ds.calibrate(
    hindcast, obs,                       # season-aware cubes, on the obs grid
    method="smoothed_regression",
    output_type="deterministic",
    temporal_sigma="constant",
    forecast_year=2024,
)
assert adjusted.dims == ("season", "lat", "lon")

# Probabilistic: below/normal/above terciles (season, tercile, lat, lon), sums to 1.
# distribution="normal" for temperature, "gamma" for precipitation. Score with `crpss`.
probs = ds.calibrate(
    hindcast, obs,
    method="smoothed_regression",
    output_type="tercile",
    distribution="gamma",
    temporal_sigma=1.5,                  # float -> cyclic Gaussian smoothing
    constrained=True,                    # analytic spread (False -> CRPS-minimizing)
    forecast_year=2024,
)
assert probs.sizes["tercile"] == 3

# Real-time: apply the hindcast fit to an out-of-sample forecast ensemble
# (e.g. OND 2026 vs a 1993-2020 hindcast). Members go through the
# hindcast-fitted coefficients, gamma parameters, and tercile boundaries.
probs_rt = ds.calibrate(
    hindcast, obs,
    method="smoothed_regression",
    output_type="tercile",
    distribution="gamma",
    forecast=forecast_members,           # (season, member, lat, lon), not in obs years
)

# Multi-model: ereg-style {model: (hindcast, forecast)} dict, but members are
# POOLED across models into one super-ensemble (reindexed member ids), not
# calibrated per model and averaged.
probs_mme = ds.calibrate(
    {"SPEAR": (spear_hcst, spear_fcst), "CanSIPS": (cansips_hcst, cansips_fcst)},
    obs,
    method="smoothed_regression",
    output_type="tercile",
    distribution="gamma",
)

# Inspect the fitted, smoothed coefficient field directly:
a = ds.seasonal_coefficients(hindcast, obs, temporal_sigma=1.5)   # (season, lat, lon)
