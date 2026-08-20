"""Tests for `deepscale.load_terrain` (Copernicus GLO-90 terrain covariates).

Everything except the marker-gated smoke test runs offline: the tile-reading
step (`deepscale.terrain._read_tile`) is monkeypatched with synthetic tiles so
the aggregation math and the cache behavior are exercised without touching the
Copernicus bucket.
"""
import numpy as np
import pytest
import xarray as xr

import deepscale as ds
from deepscale import terrain as terrain_mod


# ---------------------------------------------------------------------------
# Synthetic-tile machinery
# ---------------------------------------------------------------------------

def _synthetic_tile(url, pixels_per_degree=8):
    """A deterministic fake GLO-90 tile for the 1x1 degree cell named in `url`.

    Pixel centers sit strictly inside the tile (never on a cell edge), values
    are a smooth function of position so per-cell means/stds are nontrivial.
    """
    name = url.rsplit("/", 1)[-1]           # Copernicus_DSM_COG_30_N00_00_E010_00_DEM.tif
    parts = name.split("_")
    ns, ew = parts[4], parts[6]
    lat0 = int(ns[1:]) * (1 if ns[0] == "N" else -1)
    lon0 = int(ew[1:]) * (1 if ew[0] == "E" else -1)
    n = pixels_per_degree
    offs = (np.arange(n) + 0.5) / n
    x = lon0 + offs
    y = (lat0 + offs)[::-1]                 # descending y, like a real COG
    vals = (1000.0 * np.sin(3.0 * y[:, None]) * np.cos(2.0 * x[None, :])
            + 50.0 * y[:, None] + 20.0 * x[None, :] + 500.0)
    return x, y, vals.astype("float64")


def _expected_stats(like_lat, like_lon, tiles):
    """Reference block mean/std computed with plain loops over tile pixels."""
    lat_edges = terrain_mod._cell_edges(like_lat)
    lon_edges = terrain_mod._cell_edges(like_lon)
    ny, nx = len(like_lat), len(like_lon)
    buckets = [[[] for _ in range(nx)] for _ in range(ny)]
    for x, y, vals in tiles:
        for i, yy in enumerate(y):
            for j, xx in enumerate(x):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                iy = np.searchsorted(lat_edges, yy, side="right") - 1
                ix = np.searchsorted(lon_edges, xx, side="right") - 1
                if 0 <= iy < ny and 0 <= ix < nx:
                    buckets[iy][ix].append(v)
    mean = np.full((ny, nx), np.nan)
    std = np.full((ny, nx), np.nan)
    order_lat = np.argsort(like_lat)
    order_lon = np.argsort(like_lon)
    for iy in range(ny):
        for ix in range(nx):
            b = buckets[iy][ix]
            if b:
                mean[iy, ix] = np.mean(b)
                std[iy, ix] = np.std(b)
    # buckets are indexed on the ascending edge grid; scatter back onto the
    # possibly-descending coordinate order of `like`
    out_mean = np.full((ny, nx), np.nan)
    out_std = np.full((ny, nx), np.nan)
    for iy in range(ny):
        for ix in range(nx):
            out_mean[order_lat[iy], order_lon[ix]] = mean[iy, ix]
            out_std[order_lat[iy], order_lon[ix]] = std[iy, ix]
    return out_mean, out_std


def _like(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return xr.DataArray(
        np.zeros((lat.size, lon.size)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def fake_tiles(monkeypatch):
    """Serve synthetic tiles; record how many tile reads happened."""
    calls = {"n": 0}

    def read_tile(url):
        calls["n"] += 1
        return _synthetic_tile(url)

    monkeypatch.setattr(terrain_mod, "_read_tile", read_tile)
    return calls


@pytest.fixture
def ocean_only(monkeypatch):
    """Every tile 404s / the bucket is unreachable."""
    monkeypatch.setattr(terrain_mod, "_read_tile", lambda url: None)


# ---------------------------------------------------------------------------
# Tile URL / cell-edge helpers
# ---------------------------------------------------------------------------

def test_tile_urls_cover_region_and_format_hemispheres():
    urls = terrain_mod._tile_urls([-1.2, 1.2, -0.5, 1.5])
    names = sorted(u.rsplit("/", 1)[-1] for u in urls)
    lats = {"S02", "S01", "N00", "N01"}
    lons = {"W001", "E000", "E001"}
    expect = sorted(
        f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM.tif" for ns in lats for ew in lons
    )
    assert names == expect
    assert all(u.startswith("https://copernicus-dem-90m.s3.amazonaws.com/") for u in urls)


def test_cell_edges_regular_descending():
    edges = terrain_mod._cell_edges(np.array([2.5, 1.5, 0.5]))
    np.testing.assert_allclose(edges, [0.0, 1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Aggregation math (offline, synthetic tiles)
# ---------------------------------------------------------------------------

def test_block_mean_and_std_match_reference(fake_tiles, tmp_path):
    lat = np.array([0.25, 0.75])            # both cells inside the N00/E010 tile
    lon = np.array([10.25, 10.75])
    like = _like(lat, lon)
    out = ds.load_terrain(like, cache_dir=tmp_path)

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"elevation", "roughness"}
    assert out["elevation"].dims == ("lat", "lon")
    assert out["lat"].equals(like["lat"]) and out["lon"].equals(like["lon"])

    tiles = [_synthetic_tile(u) for u in terrain_mod._tile_urls([0.0, 1.0, 10.0, 11.0])]
    exp_mean, exp_std = _expected_stats(lat, lon, tiles)
    np.testing.assert_allclose(out["elevation"].values, exp_mean, rtol=1e-12)
    np.testing.assert_allclose(out["roughness"].values, exp_std, rtol=1e-9, atol=1e-9)
    assert "Copernicus" in out.attrs.get("source", "")


def test_multi_tile_region_and_nan_pixels(fake_tiles, monkeypatch, tmp_path):
    """A 2x2-degree region spanning 4 tiles, with NaN pixels excluded."""
    base = terrain_mod._read_tile

    def read_with_nans(url):
        x, y, vals = base(url)
        vals = vals.copy()
        vals[::3, ::3] = np.nan
        return x, y, vals

    monkeypatch.setattr(terrain_mod, "_read_tile", read_with_nans)
    lat = np.array([0.5, 1.5])
    lon = np.array([10.5, 11.5])
    like = _like(lat, lon)
    out = ds.load_terrain(like, cache_dir=tmp_path)

    tiles = [read_with_nans(u) for u in terrain_mod._tile_urls([0.0, 2.0, 10.0, 12.0])]
    exp_mean, exp_std = _expected_stats(lat, lon, tiles)
    np.testing.assert_allclose(out["elevation"].values, exp_mean, rtol=1e-12)
    np.testing.assert_allclose(out["roughness"].values, exp_std, rtol=1e-9, atol=1e-9)


def test_descending_lat_gives_same_field_on_like_coords(fake_tiles, tmp_path):
    lat_up = np.array([0.25, 0.75])
    lon = np.array([10.25, 10.75])
    up = ds.load_terrain(_like(lat_up, lon), cache_dir=tmp_path / "a")
    down = ds.load_terrain(_like(lat_up[::-1], lon), cache_dir=tmp_path / "b")

    np.testing.assert_array_equal(down["lat"].values, lat_up[::-1])
    np.testing.assert_allclose(
        down["elevation"].values, up["elevation"].values[::-1], rtol=1e-12
    )
    np.testing.assert_allclose(
        down["roughness"].values, up["roughness"].values[::-1], rtol=1e-12
    )


def test_like_may_be_a_dataset_and_builds_expected_terrain(fake_tiles, tmp_path):
    lat = np.array([0.25, 0.75])
    lon = np.array([10.25, 10.75])
    like = xr.Dataset({"precip": _like(lat, lon)})
    out = ds.load_terrain(like, cache_dir=tmp_path)
    # elevation and roughness are returned on the requested grid
    assert "elevation" in out.data_vars
    assert all(out[v].dims == ("lat", "lon") for v in out.data_vars)
    assert np.isfinite(out["elevation"].values).all()


# ---------------------------------------------------------------------------
# Cache behavior (offline)
# ---------------------------------------------------------------------------

def test_cache_roundtrip_is_offline_and_identical(fake_tiles, tmp_path):
    like = _like([0.25, 0.75], [10.25, 10.75])
    first = ds.load_terrain(like, cache_dir=tmp_path)
    n_after_first = fake_tiles["n"]
    assert n_after_first > 0
    assert list(tmp_path.glob("*.nc")), "expected a NetCDF cache file"

    second = ds.load_terrain(like, cache_dir=tmp_path)
    assert fake_tiles["n"] == n_after_first    # no tile reads on the cached path
    xr.testing.assert_allclose(first, second)
    assert second["lat"].equals(like["lat"]) and second["lon"].equals(like["lon"])


def test_cache_used_when_bucket_unreachable(monkeypatch, tmp_path):
    like = _like([0.25, 0.75], [10.25, 10.75])
    monkeypatch.setattr(terrain_mod, "_read_tile", lambda url: _synthetic_tile(url))
    first = ds.load_terrain(like, cache_dir=tmp_path)
    monkeypatch.setattr(terrain_mod, "_read_tile", lambda url: None)  # offline now
    second = ds.load_terrain(like, cache_dir=tmp_path)
    xr.testing.assert_allclose(first, second)


def test_grid_mismatch_rebuilds(fake_tiles, tmp_path):
    a = _like([0.25, 0.75], [10.25, 10.75])
    b = _like([0.125, 0.375, 0.625, 0.875], [10.25, 10.75])
    ds.load_terrain(a, cache_dir=tmp_path)
    n_after_a = fake_tiles["n"]
    out_b = ds.load_terrain(b, cache_dir=tmp_path)
    assert fake_tiles["n"] > n_after_a         # grid changed -> refetched
    assert out_b["lat"].size == 4
    assert out_b["lat"].equals(b["lat"])


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_no_tiles_no_cache_raises_clear_error(ocean_only, tmp_path):
    like = _like([0.25, 0.75], [10.25, 10.75])
    with pytest.raises(RuntimeError, match="(?i)copernicus"):
        ds.load_terrain(like, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="terrain"):
        ds.load_terrain(like, cache_dir=tmp_path)


def test_like_without_latlon_raises():
    da = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    with pytest.raises(ValueError, match="lat"):
        ds.load_terrain(da, cache_dir="unused")


def test_single_point_grid_raises():
    da = xr.DataArray(np.zeros((1, 2)), dims=["lat", "lon"],
                      coords={"lat": [0.5], "lon": [10.25, 10.75]})
    with pytest.raises(ValueError, match="at least 2"):
        ds.load_terrain(da, cache_dir="unused")


# ---------------------------------------------------------------------------
# Opt-in real-bucket smoke test (marker-gated, needs network on cold cache)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_bucket_smoke(tmp_path):
    """One land tile from the real Copernicus bucket (Mt. Kenya area)."""
    like = _like([-0.75, -0.25, 0.25, 0.75], [36.25, 36.75, 37.25, 37.75])
    try:
        out = ds.load_terrain(like, cache_dir=tmp_path)
    except RuntimeError:
        pytest.skip("Copernicus GLO-90 bucket unreachable in this environment")
    elev = out["elevation"].values
    assert np.isfinite(elev).all()
    assert elev.max() > 1500.0                 # the Kenyan highlands are in the box
    assert (out["roughness"].values >= 0).all()
