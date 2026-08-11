"""deepscale.pool_ensembles: combine per-model ensembles into one predictor cube.

Lifted from AGU's downscale_frontier.gcm_pooled — a general ensemble-prep step for feeding a
multi-model ensemble to deepscale.optimize.
"""
import numpy as np
import pytest
import xarray as xr

import deepscale
from deepscale import pool_ensembles


def _model(seed, *, years=range(1993, 2017), members=4, lat=(-4, 0, 4), lon=(34, 38, 42)):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(len(list(years)), members, len(lat), len(lon))) + 20,
        dims=("year", "member", "lat", "lon"),
        coords={"year": list(years), "member": np.arange(members),
                "lat": list(lat), "lon": list(lon)},
    )


def test_pools_members_across_models():
    das = [_model(0), _model(1), _model(2)]
    pooled = pool_ensembles(das)
    assert pooled.sizes["member"] == 12                       # 3 models x 4 members
    assert list(pooled.member.values) == list(range(12))       # contiguous, unique
    assert pooled.sizes["year"] == 24


def test_skips_none_and_requires_at_least_one():
    assert pool_ensembles([None, _model(0), None]).sizes["member"] == 4
    with pytest.raises(ValueError, match="no arrays"):
        pool_ensembles([None, None])


def test_intersects_years():
    a = _model(0, years=range(1993, 2017))
    b = _model(1, years=range(2000, 2020))
    pooled = pool_ensembles([a, b])
    assert list(pooled.year.values) == list(range(2000, 2017))


def test_regrids_onto_reference():
    a = _model(0, lat=(-4, 0, 4), lon=(34, 38, 42))
    b = _model(1, lat=(-3, 0, 3), lon=(35, 38, 41))            # different grid
    pooled = pool_ensembles([a, b])
    assert list(pooled.lat.values) == [-4, 0, 4]               # b regridded onto a
    assert not pooled.isnull().all()


def test_regrid_to_tuple_reference():
    a = _model(0, lat=(-4, 0, 4), lon=(34, 38, 42))
    b = _model(1, lat=(-3, 0, 3), lon=(35, 38, 41))            # different grid
    pooled = pool_ensembles([a, b], regrid_to=([-4, 0, 4], [34, 38, 42]))
    assert list(pooled.lat.values) == [-4, 0, 4]               # both on the tuple grid
    assert list(pooled.lon.values) == [34, 38, 42]
    assert pooled.sizes["member"] == 8


def test_regrid_to_tuple_leaves_on_grid_arrays_untouched(monkeypatch):
    """An array already on the tuple reference grid must not be re-interpolated —
    the value-equality fast path, which a DataArray.equals(<ndarray>) check skipped."""
    a = _model(0, lat=(-4, 0, 4), lon=(34, 38, 42))
    calls = {"n": 0}
    real_interp = xr.DataArray.interp

    def counting_interp(self, *args, **kwargs):
        calls["n"] += 1
        return real_interp(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "interp", counting_interp)
    pool_ensembles([a], regrid_to=([-4, 0, 4], [34, 38, 42]))
    assert calls["n"] == 0


def test_resolves_latitude_longitude_aliases():
    a = _model(0).rename({"lat": "latitude", "lon": "longitude"})
    b = _model(1).rename({"lat": "latitude", "lon": "longitude"})
    pooled = pool_ensembles([a, b])                            # would AttributeError if lat/lon hard-coded
    assert pooled.sizes["member"] == 8


def test_member_relabeling_does_not_change_optimize_score():
    """The guarantee behind rewiring AGU: optimize averages over members, so renumbering them
    (contiguous vs the old i*100 offset) must not change the result."""
    das = [_model(0), _model(1), _model(2)]
    rng = np.random.default_rng(9)
    obs = xr.DataArray(
        rng.normal(size=(24, 3, 3)) + 20, dims=("year", "lat", "lon"),
        coords={"year": list(range(1993, 2017)), "lat": [-4, 0, 4], "lon": [34, 38, 42]},
    )
    # v1: library (contiguous ids); v2: old AGU scheme (member + i*100), same data & order
    v1 = pool_ensembles([d.copy() for d in das])
    v2 = xr.concat([d.assign_coords(member=d.member + i * 100) for i, d in enumerate(das)],
                   dim="member")
    s1 = deepscale.optimize(v1, obs, methods=["cca"], primary_metric="generalized_roc",
                            verbose=False, progress=False).score
    s2 = deepscale.optimize(v2, obs, methods=["cca"], primary_metric="generalized_roc",
                            verbose=False, progress=False).score
    assert s1 == pytest.approx(s2, abs=1e-12)
