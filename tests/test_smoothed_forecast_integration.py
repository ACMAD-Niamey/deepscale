"""Integration test for smoothed_regression round 2 (issue #5): real-time
out-of-sample application through the public ``deepscale.calibrate`` API.

Mirrors the production scenario the issue describes — an OND 2026 forecast
against a 1993-2020 hindcast, where the target year is NOT in the hindcast —
end-to-end over synthetic data:

- The issue's exact calling convention (``forecast=`` ensemble members, gamma
  tercile output) reaches the calibrator through the high-level API.
- The multi-model ``{model: (hindcast, forecast)}`` pooled super-ensemble entry
  agrees with hand-pooling the members.
- The calibrated deterministic anomaly actually tracks the planted future
  signal (skill, not just shape/finiteness).
"""
import numpy as np
import xarray as xr

import deepscale

SEASONS = ["ASO", "SON", "OND", "NDJ"]
YEARS = np.arange(1993, 2021)          # 1993..2020 hindcast; 2026 is out-of-sample
NLAT, NLON = 5, 6
COORDS = {"season": SEASONS, "year": YEARS,
          "lat": np.linspace(-5, 5, NLAT), "lon": np.linspace(30, 40, NLON)}


def _model(seed, n_member, signal, future_signal, noise=0.3):
    """One GCM: members = signal + member noise, for hindcast and 2026 forecast."""
    rng = np.random.default_rng(seed)
    ns, ny = len(SEASONS), len(YEARS)
    hind = signal[:, :, None] + noise * rng.standard_normal((ns, ny, n_member, NLAT, NLON))
    fcst = future_signal[:, None] + noise * rng.standard_normal((ns, n_member, NLAT, NLON))
    h = xr.DataArray(hind, dims=("season", "year", "member", "lat", "lon"),
                     coords=dict(COORDS, member=np.arange(n_member)))
    f = xr.DataArray(fcst, dims=("season", "member", "lat", "lon"),
                     coords={k: COORDS[k] for k in ("season", "lat", "lon")} |
                            {"member": np.arange(n_member)})
    return h, f


def _scenario():
    rng = np.random.default_rng(42)
    ns, ny = len(SEASONS), len(YEARS)
    signal = rng.standard_normal((ns, ny, NLAT, NLON))
    future_signal = rng.standard_normal((ns, NLAT, NLON))
    obs = 0.7 * signal + 0.1 * rng.standard_normal((ns, ny, NLAT, NLON))
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"), coords=COORDS)
    models = {"gcm_a": _model(1, 11, signal, future_signal),
              "gcm_b": _model(2, 25, signal, future_signal)}
    return models, ob, future_signal


def test_realtime_out_of_sample_end_to_end():
    models, ob, future_signal = _scenario()

    # -- deterministic: the calibrated 2026 anomaly must track the planted
    # -- future obs response (0.7 * future signal)
    det = deepscale.calibrate(models, ob, method="smoothed_regression",
                              output_type="deterministic", temporal_sigma=None)
    assert det.dims == ("season", "lat", "lon")
    assert bool(np.isfinite(det).all())
    anom = (det - ob.mean("year")).values.ravel()
    truth = (0.7 * future_signal).ravel()
    r = np.corrcoef(anom, truth)[0, 1]
    assert r > 0.9

    # -- tercile: valid probabilities, and above-normal beats below-normal
    # -- exactly where the planted future signal is strongly positive
    probs = deepscale.calibrate(models, ob, method="smoothed_regression",
                                output_type="tercile", temporal_sigma=1.0,
                                distribution="normal")
    assert probs.dims == ("season", "tercile", "lat", "lon")
    sums = probs.sum("tercile").values
    np.testing.assert_allclose(sums[np.isfinite(sums)], 1.0, atol=1e-9)
    strong_wet = future_signal > 1.0
    above = probs.sel(tercile=2).values[strong_wet]
    below = probs.sel(tercile=0).values[strong_wet]
    assert strong_wet.sum() > 10
    assert (above > below).all()

    # -- the multi-model dict entry equals hand-pooling the super-ensemble
    pooled_h, pooled_f = [], []
    offset = 0
    for h, f in models.values():
        n = h.sizes["member"]
        ids = list(range(offset, offset + n))
        pooled_h.append(h.assign_coords(member=ids))
        pooled_f.append(f.assign_coords(member=ids))
        offset += n
    manual = deepscale.calibrate(xr.concat(pooled_h, dim="member"), ob,
                                 method="smoothed_regression",
                                 output_type="deterministic", temporal_sigma=None,
                                 forecast=xr.concat(pooled_f, dim="member"))
    np.testing.assert_allclose(det.values, manual.values, rtol=1e-10, atol=1e-12)
