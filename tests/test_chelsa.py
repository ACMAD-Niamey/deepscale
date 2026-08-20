import numpy as np
import xarray as xr

import pytest

from deepscale.methods.chelsa import CHELSAMethod, chelsa_precipitation, wind_effect
from deepscale.registry import get_method
from deepscale import downscale


def _field(values, lat=None, lon=None):
    values = np.asarray(values, dtype=float)
    lat = np.arange(values.shape[0], dtype=float) * 0.05 if lat is None else lat
    lon = np.arange(values.shape[1], dtype=float) * 0.05 if lon is None else lon
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_flat_terrain_has_unit_wind_effect():
    z = _field(np.zeros((9, 9)))
    u = xr.ones_like(z)
    v = xr.zeros_like(z)
    actual = wind_effect(z, u, v, max_distance_km=20)
    np.testing.assert_allclose(actual, 1.0)


def test_crosswind_ridge_creates_distinct_windward_and_leeward_pattern():
    z = _field(np.tile(np.array([0, 0, 100, 300, 700, 300, 100, 0, 0]), (9, 1)))
    effect = wind_effect(z, xr.ones_like(z), xr.zeros_like(z), max_distance_km=30)
    assert np.nanstd(effect.values) > 0
    assert not np.allclose(effect[:, :4], effect[:, -4:])


def test_parent_cell_means_are_exactly_conserved():
    coarse = _field([[2.0, 4.0], [6.0, 8.0]], lat=[0.1, 0.3], lon=[0.1, 0.3])
    fine_lat = np.arange(0.025, 0.4, 0.05)
    fine_lon = np.arange(0.025, 0.4, 0.05)
    z = _field(np.add.outer(fine_lat, fine_lon) * 5000, fine_lat, fine_lon)
    result = chelsa_precipitation(coarse, z, xr.ones_like(z), xr.zeros_like(z),
                                  max_distance_km=20)
    iy = np.abs(fine_lat[:, None] - coarse.lat.values).argmin(axis=1)
    ix = np.abs(fine_lon[:, None] - coarse.lon.values).argmin(axis=1)
    for y in range(2):
        for x in range(2):
            cells = result.values[(iy[:, None] == y) & (ix[None, :] == x)]
            np.testing.assert_allclose(cells.mean(), coarse.values[y, x], rtol=1e-12)


def test_pbl_inputs_are_all_or_nothing():
    coarse = _field([[2.0, 2.0], [2.0, 2.0]], lat=[0.1, 0.3], lon=[0.1, 0.3])
    z = _field(np.zeros((8, 8)), np.arange(8) * 0.05, np.arange(8) * 0.05)
    try:
        chelsa_precipitation(coarse, z, xr.ones_like(z), xr.zeros_like(z),
                             boundary_layer_height=xr.ones_like(z))
    except ValueError as exc:
        assert "PBL correction" in str(exc)
    else:
        raise AssertionError("expected missing coarse_orography to fail")


def test_registry_exposes_paper_defined_method():
    assert get_method("chelsa") is CHELSAMethod


def test_published_pbl_equation_and_default_offset(monkeypatch):
    """Karger 2021: H_B = H / (1 - (distance-zmax)/9000)."""
    coarse = _field([[2.0]], lat=[0.1], lon=[0.1])
    fine = _field([[0.0, 1000.0], [2000.0, 3000.0]], [0.05, 0.15], [0.05, 0.15])
    pbl = xr.zeros_like(fine)
    coarse_orog = xr.zeros_like(fine)
    monkeypatch.setattr(
        "deepscale.methods.chelsa.wind_effect",
        lambda elevation, u, v, **kwargs: xr.ones_like(elevation),
    )
    actual = chelsa_precipitation(
        coarse,
        fine,
        xr.ones_like(fine),
        xr.zeros_like(fine),
        boundary_layer_height=pbl,
        coarse_orography=coarse_orog,
    )
    distance = np.abs(fine.values - 500.0)
    maximum = distance.max()
    intensity = 1.0 / (1.0 - (distance - maximum) / 9000.0)
    expected = 2.0 * intensity / intensity.mean()
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(actual.mean(), 2.0, rtol=1e-12)


def test_method_preserves_members_and_uses_training_year_atmosphere():
    coarse_lat = [0.1, 0.3]
    coarse_lon = [0.1, 0.3]
    years = [2000, 2001]
    hindcast = xr.DataArray(
        np.ones((2, 2, 2, 2)),
        dims=("year", "member", "lat", "lon"),
        coords={"year": years, "member": [0, 1], "lat": coarse_lat, "lon": coarse_lon},
    )
    fine_lat = np.arange(0.025, 0.4, 0.05)
    fine_lon = np.arange(0.025, 0.4, 0.05)
    terrain = _field(np.add.outer(fine_lat, fine_lon) * 5000, fine_lat, fine_lon)
    obs = xr.concat([terrain * 0 + 1, terrain * 0 + 1], dim="year").assign_coords(year=years)
    wind = xr.concat([xr.ones_like(terrain), xr.ones_like(terrain) * 3], dim="year").assign_coords(year=years)

    method = CHELSAMethod()
    method.fit(
        hindcast.sel(year=[2000]),
        obs.sel(year=[2000]),
        terrain=terrain,
        u_wind=wind,
        v_wind=xr.zeros_like(wind),
        max_distance_km=20,
    )
    assert method.u_wind_.dims == ("lat", "lon")
    np.testing.assert_allclose(method.u_wind_, 1.0)
    result = method.predict(hindcast.sel(year=2001, drop=True))
    assert result.dims == ("member", "lat", "lon")
    assert result.sizes["member"] == 2


def test_forecast_atmosphere_must_be_preselected_2d():
    method = CHELSAMethod()
    method.elevation_ = _field(np.zeros((4, 4)))
    method.u_wind_ = xr.ones_like(method.elevation_)
    method.v_wind_ = xr.zeros_like(method.elevation_)
    method.boundary_layer_height_ = None
    method.coarse_orography_ = None
    method.exposure_ = None
    method.pbl_offset_m_ = 500.0
    method.height_scale_m_ = 9000.0
    method.max_distance_km_ = 20.0
    coarse = _field([[1.0, 1.0], [1.0, 1.0]], lat=[0.05, 0.15], lon=[0.05, 0.15])
    atmosphere = xr.Dataset(
        {
            "u_wind": xr.ones_like(method.elevation_).expand_dims(year=[2001]),
            "v_wind": xr.zeros_like(method.elevation_).expand_dims(year=[2001]),
        }
    )
    with pytest.raises(ValueError, match="select the forecast period"):
        method.predict(coarse, forecast_atmosphere=atmosphere)


def test_downscale_public_api_runs_chelsa():
    years = [2000, 2001]
    coarse = xr.DataArray(
        np.ones((2, 1, 2, 2)),
        dims=("year", "member", "lat", "lon"),
        coords={"year": years, "member": [0], "lat": [0.1, 0.3], "lon": [0.1, 0.3]},
    )
    fine = _field(np.zeros((8, 8)), np.arange(8) * 0.05, np.arange(8) * 0.05)
    obs = xr.concat([fine + 1, fine + 1], dim="year").assign_coords(year=years)
    result = downscale(
        coarse,
        obs,
        method="chelsa",
        terrain=fine,
        u_wind=xr.ones_like(obs),
        v_wind=xr.zeros_like(obs),
        max_distance_km=20,
        verbose=False,
    )
    assert result.dims == ("member", "lat", "lon")
    assert result.shape == (1, 8, 8)


def test_time_atmosphere_is_subset_to_training_years():
    field = _field(np.zeros((4, 4)))
    time = np.array(["2000-03-01", "2001-03-01"], dtype="datetime64[ns]")
    wind = xr.concat([field + 1, field + 9], dim="time").assign_coords(time=time)
    hindcast = xr.DataArray(
        np.ones((1, 1, 1, 1)),
        dims=("year", "member", "lat", "lon"),
        coords={"year": [2000], "member": [0], "lat": [0.1], "lon": [0.1]},
    )
    obs = (field + 1).expand_dims(year=[2000])
    method = CHELSAMethod()
    method.fit(hindcast, obs, terrain=field, u_wind=wind, v_wind=wind * 0)
    np.testing.assert_allclose(method.u_wind_, 1.0)


def test_checkpoint_roundtrip_preserves_chelsa_prediction(tmp_path):
    coarse = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("member", "lat", "lon"),
        coords={"member": [0], "lat": [0.1, 0.3], "lon": [0.1, 0.3]},
    )
    fine = _field(np.add.outer(np.arange(8), np.arange(8)) * 20.0)
    method = CHELSAMethod()
    method.elevation_ = fine
    method.u_wind_ = xr.ones_like(fine)
    method.v_wind_ = xr.zeros_like(fine)
    method.boundary_layer_height_ = None
    method.coarse_orography_ = None
    method.exposure_ = None
    method.pbl_offset_m_ = 500.0
    method.height_scale_m_ = 9000.0
    method.max_distance_km_ = 20.0
    expected = method.predict(coarse)
    checkpoint = tmp_path / "chelsa.pkl"
    method.save(checkpoint)
    restored = CHELSAMethod().load(checkpoint)
    xr.testing.assert_identical(restored.predict(coarse), expected)


@pytest.mark.parametrize("bad", [-1.0, -0.01])
def test_rejects_negative_precipitation(bad):
    coarse = _field([[bad]], lat=[0.1], lon=[0.1])
    fine = _field(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        chelsa_precipitation(coarse, fine, xr.ones_like(fine), xr.zeros_like(fine))
