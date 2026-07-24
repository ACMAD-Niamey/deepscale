"""Kharin et al. (2017) smoothed seasonal regression calibration, LOYO-scored.

The one-call path is the registered calibrator:
    ds.calibrate(fc, obs, method="smoothed_regression", output_type="tercile",
                 temporal_sigma=1.0, distribution="gamma")
That is fit-and-apply on the hindcast; CV scoring is the caller's concern. This
example uses the underlying function layer to run an honest leave-one-year-out
loop: calibrate mean AND spread per gridpoint, smooth the coefficients across
the seasonal cycle, and score with CRPSS against climatology. Rainfall is
regressed in gamma->normal space (a gamma variable is skewed; regression
assumes ~normal).

Inputs are a rectangular (season, year, member, lat, lon) forecast cube and a
(season, year, lat, lon) obs cube on the same grid — build them by intersecting
the years available in every season (wraparound seasons like DJF otherwise
NaN-pad a union of years).
"""

import numpy as np
import xarray as xr

import deepscale as ds
from deepscale.cv import loyo
from deepscale.methods import smoothed_regression as sr
from deepscale.metrics import crpss as cs

# fc: (season, year, member, lat, lon) precip, mm/day; obs: (season, year, lat, lon)
years = fc.year.values

# --- 0. Mask cells too dry for a gamma fit (gamma undefined where it never rains)
wet = obs.mean(["season", "year"]) >= 0.5          # mm/day, ~GHACOF-style dry mask
fc, obs = fc.where(wet), obs.where(wet)

# --- 1. Deterministic slope (exported at top level), three smoothing settings ---
for sigma in (None, 1.0, "constant"):              # per-season | cyclic Gaussian | pooled
    a = ds.seasonal_coefficients(fc, obs, temporal_sigma=sigma)   # (season, lat, lon)

# --- 2. Probabilistic (mean + spread scaling), honest LOYO loop ------------------
crps_f, crps_c = [], []
for train_years, test_year in loyo(years):
    tr_fc, tr_ob = fc.sel(year=train_years), obs.sel(year=train_years)

    # gamma->normal transform fitted on training obs only, per (season, lat, lon)
    shp = xr.apply_ufunc(lambda x: sr.fit_gamma(x)[0], tr_ob,
                         input_core_dims=[["year"]], vectorize=True)
    scl = xr.apply_ufunc(lambda x: sr.fit_gamma(x)[1], tr_ob,
                         input_core_dims=[["year"]], vectorize=True)
    to_n = lambda x: xr.apply_ufunc(sr.gamma_to_normal, x, shp, scl)

    mu_tr = to_n(tr_fc).mean("member") - to_n(tr_fc).mean(["member", "year"])
    sg_tr = to_n(tr_fc).std("member")
    oa_tr = to_n(tr_ob) - to_n(tr_ob).mean("year")

    a, b = sr.fit_ab_field(mu_tr.transpose("season", "year", "lat", "lon").values,
                           sg_tr.transpose("season", "year", "lat", "lon").values,
                           oa_tr.transpose("season", "year", "lat", "lon").values,
                           constrained=True)
    a_s, b_s = sr.smooth_ab(a, b, 1.0)             # None | float sigma | "constant"

    ho = fc.sel(year=test_year)
    mu_ho = to_n(ho).mean("member") - to_n(tr_fc).mean(["member", "year"])
    sg_ho = to_n(ho).std("member")
    oa_ho = to_n(obs.sel(year=test_year)) - to_n(tr_ob).mean("year")

    crps_f.append(cs.crps_normal(a_s * mu_ho.values, b_s * sg_ho.values, oa_ho.values))
    crps_c.append(cs.crps_climatology(oa_ho.values, oa_tr.std("year").values))

score = cs.crpss(np.nanmean(np.stack(crps_f)), np.nanmean(np.stack(crps_c)))
print(f"CRPSS vs climatology: {score:.3f}")        # > 0 means beats climatology

# The registered metric form takes an xr.Dataset of anomaly mu/sigma instead:
#   ds.skill(xr.Dataset({"mu": mu_cal, "sigma": sig_cal}), obs_anom, metrics="crpss")
