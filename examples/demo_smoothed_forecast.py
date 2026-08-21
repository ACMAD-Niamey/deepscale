"""
Demo: smoothed_regression round 2 — real-time out-of-sample application.

The Kharin et al. (2017) smoothed-regression calibrator fits its coefficients
on the hindcast; round 2 (issue #5) lets it APPLY that fit to a forecast
ensemble whose target year is not in the hindcast — the real-time workflow
(e.g. OND 2026 against a 1993-2020 hindcast):

    calibrate(hindcast, obs, method="smoothed_regression",
              forecast=forecast_members,           # (season, member, lat, lon)
              output_type="tercile", distribution="gamma")

Also shown: the multi-model {model: (hindcast, forecast)} entry, which pools
members across models into ONE super-ensemble (the Kharin experiment design)
instead of calibrating per model and averaging like eReg.

Network-free - uses small synthetic rainfall-like data.
Run from the repository root:

    uv run python examples/demo_smoothed_forecast.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr

import deepscale as ds

SEASONS = ["ASO", "SON", "OND", "NDJ"]
YEARS = np.arange(1993, 2021)                       # hindcast window
NLAT, NLON = 5, 6
COORDS = {"season": SEASONS, "year": YEARS,
          "lat": np.linspace(-5, 5, NLAT), "lon": np.linspace(30, 40, NLON)}
TERCILE_NAMES = {0: "below", 1: "near", 2: "above"}


def _synthetic_models():
    """Two GCMs sharing a planted rainfall signal, plus a wet-leaning 2026
    forecast that the hindcast never saw."""
    rng = np.random.default_rng(0)
    ns, ny = len(SEASONS), len(YEARS)
    base = rng.gamma(2.0, 50.0, (ns, ny, NLAT, NLON))            # shared "truth"
    future = rng.gamma(2.0, 50.0, (ns, NLAT, NLON)) * 1.4        # wet 2026
    obs = xr.DataArray(base * rng.gamma(8.0, 1 / 8.0, base.shape),
                       dims=("season", "year", "lat", "lon"), coords=COORDS,
                       name="precip")

    models = {}
    for name, seed, n_member in (("gcm_a", 1, 11), ("gcm_b", 2, 25)):
        r = np.random.default_rng(seed)
        hind = base[:, :, None] * r.gamma(8.0, 1 / 8.0, (ns, ny, n_member, NLAT, NLON))
        fcst = future[:, None] * r.gamma(8.0, 1 / 8.0, (ns, n_member, NLAT, NLON))
        h = xr.DataArray(hind, dims=("season", "year", "member", "lat", "lon"),
                         coords=dict(COORDS, member=np.arange(n_member)))
        f = xr.DataArray(fcst, dims=("season", "member", "lat", "lon"),
                         coords={k: COORDS[k] for k in ("season", "lat", "lon")} |
                                {"member": np.arange(n_member)})
        models[name] = (h, f)
    return models, obs


def main() -> None:
    models, obs = _synthetic_models()
    n_members = sum(h.sizes["member"] for h, _f in models.values())
    print(f"hindcast {YEARS[0]}-{YEARS[-1]}, target 2026 (out-of-sample), "
          f"{len(models)} models pooled into {n_members} members\n")

    # --- tercile probabilities for the real-time forecast (gamma: rainfall) ---
    probs = ds.calibrate(models, obs, method="smoothed_regression",
                         output_type="tercile", distribution="gamma",
                         temporal_sigma=1.0)
    print(f"tercile probabilities: dims={probs.dims}, "
          f"out_of_sample={probs.attrs['out_of_sample']}")
    for s in SEASONS:
        mean_p = probs.sel(season=s).mean(["lat", "lon"])
        parts = ", ".join(f"{TERCILE_NAMES[t]} {float(mean_p.sel(tercile=t)):.2f}"
                          for t in (0, 1, 2))
        print(f"  {s}: domain-mean P({parts})")
    dominant = int(probs.mean(["season", "lat", "lon"]).argmax("tercile"))
    print(f"  wet-leaning forecast detected: dominant category = "
          f"{TERCILE_NAMES[dominant]!r}\n")

    # --- deterministic calibrated field for the same target ------------------
    det = ds.calibrate(models, obs, method="smoothed_regression",
                       output_type="deterministic", temporal_sigma=1.0)
    clim = obs.mean("year")
    print(f"deterministic field: dims={det.dims}")
    print(f"  domain-mean calibrated 2026 rainfall: {float(det.mean()):7.1f}")
    print(f"  domain-mean obs climatology:          {float(clim.mean()):7.1f}")
    print(f"  anomaly (wet 2026 planted at +40%):   {float((det - clim).mean()):+7.1f}\n")

    # --- the same fit still targets hindcast years (round 1, retro) ----------
    # forecast= and forecast_year= are mutually exclusive, so retro-forecast
    # targeting takes the hindcasts alone
    hindcasts_only = {name: h for name, (h, _f) in models.items()}
    retro = ds.calibrate(hindcasts_only, obs, method="smoothed_regression",
                         output_type="tercile", distribution="gamma",
                         temporal_sigma=1.0, forecast_year=2015)
    print(f"retro-forecast 2015 (hindcast-mode) still works: dims={retro.dims}, "
          f"out_of_sample={retro.attrs['out_of_sample']}")


if __name__ == "__main__":
    main()
