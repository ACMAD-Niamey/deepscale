import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 15. Plotting subpackage
# ===================================================================

def test_plotting_package_imports():
    """Package must import cleanly even when matplotlib/cartopy aren't installed."""
    import deepscale.plotting  # noqa: F401


def test_plot_skill_maps_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.skill import SkillReport
    from deepscale.plotting.skill import plot_skill_maps

    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    rpss = xr.DataArray(
        np.random.RandomState(0).uniform(-1, 1, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    rmse = xr.DataArray(
        np.random.RandomState(1).uniform(0, 2, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    report = SkillReport(scores={"rpss": float(rpss.mean()), "rmse": float(rmse.mean())},
                         spatial={"rpss": rpss, "rmse": rmse})

    fig = plot_skill_maps(report, ["rpss", "rmse"], ncols=2)

    assert fig is not None
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_domains_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.domains import plot_domains

    # predictand: East Africa, predictor: tropical Pacific (antimeridian-spanning)
    fig = plot_domains(
        predictor_extent=(-20, 20, 120, -60),     # lon_w > lon_e — crosses dateline
        predictand_extent=(-12, 15, 22, 52),
    )

    assert fig is not None
    plt.close(fig)


def test_plot_tercile_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast

    n_lat, n_lon = 4, 5
    probs = np.zeros((3, n_lat, n_lon))
    probs[0, :, :] = 0.15
    probs[1, :, :] = 0.25
    probs[2, :, :] = 0.60
    pr_fcst = xr.DataArray(
        probs,
        dims=["tercile", "lat", "lon"],
        coords={
            "tercile": [0, 1, 2],
            "lat": np.linspace(-5, 5, n_lat),
            "lon": np.linspace(30, 45, n_lon),
        },
    )
    fig = plot_tercile_forecast(pr_fcst)
    assert fig is not None
    plt.close(fig)


def test_tercile_rgb_leaves_masked_cells_blank():
    """NaN-masked cells (significance mask / uncalibratable) must render blank
    (white), not be painted into a confident below/above category, and must not
    leak NaN into the RGB image."""
    from deepscale.plotting.forecasts import _tercile_rgb

    probs = np.full((3, 1, 2), np.nan)
    probs[:, 0, 0] = [0.7, 0.2, 0.1]          # confident below-normal (cat 0)
    # cell (0, 1) stays all-NaN -> masked / no valid forecast

    rgb = _tercile_rgb(probs, red_cat=0, blue_cat=2)

    np.testing.assert_array_equal(rgb[0, 1], [1.0, 1.0, 1.0])   # blank, not red
    assert rgb[0, 0, 0] == 1.0 and rgb[0, 0, 1] < 1.0           # valid cell is red
    assert np.isfinite(rgb).all()                               # no NaN pixels


def test_plot_tercile_forecast_accepts_latitude_longitude_dims():
    """calibrate()/logistic_forecast() preserve obs dim names; a (tercile,
    latitude, longitude) forecast must plot without a transpose/attr error."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast

    n_lat, n_lon = 4, 5
    probs = np.full((3, n_lat, n_lon), 1.0 / 3.0)
    pr = xr.DataArray(
        probs,
        dims=["tercile", "latitude", "longitude"],
        coords={
            "tercile": [0, 1, 2],
            "latitude": np.linspace(-5, 5, n_lat),
            "longitude": np.linspace(30, 45, n_lon),
        },
    )
    fig = plot_tercile_forecast(pr)
    assert fig is not None
    plt.close(fig)


def test_to_0_360_shifts_western_hemisphere_geometry():
    """The geopandas basemap fallback must shift -180..180 shapefile geometries
    into the 0-360 convention so coastlines align with 0-360 forecast grids."""
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    import geopandas as gpd
    from shapely.geometry import LineString

    from deepscale.plotting.forecasts import _to_0_360

    west = LineString([(-100.0, 0.0), (-90.0, 10.0)])   # western hemisphere
    east = LineString([(30.0, 0.0), (40.0, 10.0)])       # eastern hemisphere
    gdf = gpd.GeoDataFrame(geometry=[west, east])

    shifted = _to_0_360(gdf)

    assert shifted.geometry.iloc[0].bounds[0] >= 180.0   # west moved into 0-360
    assert shifted.geometry.iloc[1].bounds[0] == 30.0    # east left untouched


def test_tercile_codes_maps_dominant_category_and_bin():
    from deepscale.plotting.forecasts import _tercile_codes
    prob_bins = [33.3, 40, 50, 60, 70, 100.01]   # n = 5 bins
    # one above-dominant cell at 65% (bin index 3), one below-dominant at 45% (bin 1)
    probs = np.array([
        [[0.20, 0.45]],   # below
        [[0.15, 0.30]],   # normal
        [[0.65, 0.25]],   # above
    ], dtype=float)
    code, valid = _tercile_codes(probs, prob_bins)
    assert valid.all()
    # above base 0 + bin 3 = 3 ; below base 2*5=10 + bin 1 = 11
    assert code[0, 0] == 3
    assert code[0, 1] == 11


def test_tercile_codes_marks_all_nan_cell_invalid():
    from deepscale.plotting.forecasts import _tercile_codes
    probs = np.full((3, 1, 1), np.nan)
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    assert not valid[0, 0]
    assert code[0, 0] == -1


def _ghacof_style():
    from deepscale.plotting import TercileStyle
    return TercileStyle(
        below_colors=["#fcf3c8", "#fae678", "#f8d808", "#e6b400", "#d49e00"],
        normal_colors=["#eefcff", "#e7f8f8", "#d6efef", "#c7e8e8", "#c7e8e8"],
        above_colors=["#c9f5c2", "#38f838", "#34c818", "#38a808", "#188c08"],
        prob_bins=[33.3, 40, 50, 60, 70, 100.01],
    )


def test_plot_terciles_styled_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast
    lat = np.linspace(-5, 5, 6); lon = np.linspace(30, 45, 8)
    probs = np.zeros((3, 6, 8)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    pr = xr.DataArray(probs, dims=["tercile", "lat", "lon"],
                      coords={"tercile": ["below", "normal", "above"], "lat": lat, "lon": lon})
    fig = plot_tercile_forecast(pr, style=_ghacof_style(), legend=True, title="styled")
    assert fig is not None
    plt.close(fig)


def test_plot_field_smoke_returns_mappable():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from deepscale.plotting.forecasts import plot_field
    lat = np.linspace(-5, 5, 6); lon = np.linspace(30, 45, 8)
    field = xr.DataArray(np.linspace(-40, 40, 48).reshape(6, 8),
                         dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    im = plot_field(field, style=_ghacof_style(), ax=ax, cmap="BrBG",
                    vmin=-40, vmax=40, title="difference")
    assert im is not None and hasattr(im, "get_array")   # a Matplotlib mappable
    plt.close(fig)


def test_plot_field_honors_nodata_color():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from deepscale.plotting.forecasts import plot_field
    from deepscale.plotting import TercileStyle
    lat = np.linspace(-5, 5, 4); lon = np.linspace(30, 40, 4)
    field = xr.DataArray(np.zeros((4, 4)), dims=["lat", "lon"],
                         coords={"lat": lat, "lon": lon})
    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3, 40, 50, 60, 70, 100.01], nodata_color="#123456")
    fig, ax = plt.subplots()   # plain axes: exercises the non-geo path
    im = plot_field(field, style=style, ax=ax, cmap="BrBG")
    assert im.get_cmap().get_bad() == pytest.approx(to_rgba("#123456"))
    plt.close(fig)


def _tercile_da(above, lat, lon):
    p = np.zeros((3, len(lat), len(lon)))
    p[2] = above; p[1] = 0.3; p[0] = 1.0 - above - 0.3
    return xr.DataArray(p, dims=["tercile", "lat", "lon"],
                        coords={"tercile": ["below", "normal", "above"], "lat": lat, "lon": lon})


def test_plot_tercile_comparison_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from deepscale.plotting.forecasts import plot_tercile_comparison
    lat = np.linspace(-5, 5, 6); lon = np.linspace(30, 45, 8)
    fc, ref = _tercile_da(0.5, lat, lon), _tercile_da(0.4, lat, lon)
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": ccrs.PlateCarree()})
    out_axes, diff_im = plot_tercile_comparison(
        fc, ref, style=_ghacof_style(), axes=axes,
        labels=("A", "B", "A - B"), diff_cmap="BrBG", diff_limit=40)
    assert len(out_axes) == 3
    assert diff_im is not None and hasattr(diff_im, "get_array")   # mappable for a colorbar
    assert axes[2].get_title() == "A - B"
    plt.close(fig)


def test_plot_tercile_comparison_regrids_reference_on_different_grid():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from deepscale.plotting.forecasts import plot_tercile_comparison
    fc = _tercile_da(0.5, np.linspace(-5, 5, 6), np.linspace(30, 45, 8))     # coarse
    ref = _tercile_da(0.4, np.linspace(-5, 5, 12), np.linspace(30, 45, 16))  # finer grid
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": ccrs.PlateCarree()})
    _, diff_im = plot_tercile_comparison(fc, ref, axes=axes)   # must regrid ref -> fc, not error
    assert diff_im is not None
    plt.close(fig)


def test_region_masks_dry_and_clip():
    pytest.importorskip("shapely")
    from deepscale.plotting.forecasts import _region_masks
    from deepscale.plotting import TercileStyle
    lat = np.array([0.0, 1.0]); lon = np.array([37.0, 200.0])   # 37E in Kenya, 200E mid-Pacific
    dry = np.zeros((2, 2), dtype=bool); dry[0, 0] = True
    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3, 40, 50, 60, 70, 100.01], dry_mask=dry, clip_to=["Kenya"])
    dry_out, outside = _region_masks(lat, lon, style)
    assert dry_out[0, 0] and not dry_out[1, 1]     # dry mask preserved
    assert outside[0, 1]                            # 200E ocean, outside Kenya
    assert not outside[0, 0]                        # 37E / 0N is inside Kenya


def test_apply_style_masks_dry_and_clip():
    pytest.importorskip("shapely")
    import numpy as np
    from deepscale.plotting.forecasts import _apply_style_masks, _tercile_codes
    from deepscale.plotting import TercileStyle
    lat = np.array([0.0, 1.0]); lon = np.array([37.0, 200.0])   # 37E in Kenya, 200E mid-Pacific
    probs = np.zeros((3, 2, 2)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    dry = np.zeros((2, 2), dtype=bool); dry[0, 0] = True
    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3,40,50,60,70,100.01], dry_mask=dry, clip_to=["Kenya"])
    out = _apply_style_masks(code.copy(), lat, lon, style)
    assert out[0, 0] == 15          # dry code = 3*5
    assert out[0, 1] == -1          # 200E is ocean, outside Kenya -> masked


def test_apply_style_masks_raises_on_unknown_country():
    pytest.importorskip("shapely")
    pytest.importorskip("cartopy")
    from deepscale.plotting.forecasts import _country_geometry
    with pytest.raises(ValueError):
        _country_geometry(["Nonexististan"])


def test_apply_style_masks_clip_wins_over_dry():
    """A cell that is both dry and outside the clip geometry must end up -1:
    the clip is applied after the dry paint, not before."""
    pytest.importorskip("shapely")
    pytest.importorskip("cartopy")
    import numpy as np
    from deepscale.plotting.forecasts import _apply_style_masks, _tercile_codes
    from deepscale.plotting import TercileStyle
    lat = np.array([0.0, 1.0]); lon = np.array([37.0, 200.0])   # 37E in Kenya, 200E mid-Pacific (ocean)
    probs = np.zeros((3, 2, 2)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    dry = np.zeros((2, 2), dtype=bool); dry[0, 1] = True   # 200E marked dry, but it's outside Kenya
    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3,40,50,60,70,100.01], dry_mask=dry, clip_to=["Kenya"])
    out = _apply_style_masks(code.copy(), lat, lon, style)
    assert out[0, 1] == -1          # clip wins over dry, not 15


def test_apply_style_masks_aligns_dataarray_mask_on_different_grid():
    """A coordinate-bearing dry_mask on a DIFFERENT grid than the plotted field
    must be aligned by coordinate value (nearest), not raw positional indexing.

    Regression test for IndexError: boolean index did not match indexed array,
    raised when a dry_mask built on one grid was applied to a field on another.
    """
    pytest.importorskip("shapely")
    from deepscale.plotting.forecasts import _apply_style_masks, _tercile_codes
    from deepscale.plotting import TercileStyle

    # Mask grid: finer resolution than the field, lat -10..10 step 1. lon_mask
    # spans wider than the field's lon so nearest-neighbor stays in-bounds.
    lat_mask = np.arange(-10, 11, 1, dtype=float)
    lon_mask = np.array([20.0, 30.0, 40.0, 50.0])
    mask_vals = np.broadcast_to((lat_mask >= 2.0)[:, None], (lat_mask.size, lon_mask.size))
    dry_mask = xr.DataArray(mask_vals, dims=["lat", "lon"],
                            coords={"lat": lat_mask, "lon": lon_mask})

    # Field grid: coarser resolution AND lon-offset from the mask grid -> shapes differ.
    lat = np.arange(-10, 11, 2, dtype=float)
    lon = np.array([32.0, 42.0])
    probs = np.zeros((3, lat.size, lon.size))
    probs[2] = 0.9; probs[1] = 0.07; probs[0] = 0.03
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    assert dry_mask.shape != code.shape   # confirm the grids genuinely differ

    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3,40,50,60,70,100.01], dry_mask=dry_mask, clip_to=None)

    out = _apply_style_masks(code.copy(), lat, lon, style)   # must not raise IndexError

    for i, la in enumerate(lat):
        for j in range(lon.size):
            if la >= 2.0:
                assert out[i, j] == 15   # geographically dry -> painted dry (3*n, n=5)
            else:
                assert out[i, j] != 15   # geographically not dry -> left as tercile code


def test_apply_style_masks_ndarray_shape_mismatch_raises():
    """A bare ndarray dry_mask has no coordinates to align by, so a shape
    mismatch against the plotted field must raise a clear error instead of
    a raw positional-indexing IndexError."""
    pytest.importorskip("shapely")
    from deepscale.plotting.forecasts import _apply_style_masks, _tercile_codes
    from deepscale.plotting import TercileStyle
    lat = np.array([0.0, 1.0, 2.0]); lon = np.array([37.0, 40.0])
    probs = np.zeros((3, 3, 2)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    code, valid = _tercile_codes(probs, [33.3, 40, 50, 60, 70, 100.01])
    dry = np.zeros((2, 2), dtype=bool)   # wrong shape: field is (3, 2)
    style = TercileStyle(below_colors=["a"]*5, normal_colors=["a"]*5, above_colors=["a"]*5,
                         prob_bins=[33.3,40,50,60,70,100.01], dry_mask=dry, clip_to=None)
    with pytest.raises(ValueError):
        _apply_style_masks(code.copy(), lat, lon, style)


def test_plot_terciles_no_style_still_works():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast
    lat = np.linspace(-5, 5, 6); lon = np.linspace(30, 45, 8)
    probs = np.zeros((3, 6, 8)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    pr = xr.DataArray(probs, dims=["tercile", "lat", "lon"],
                      coords={"tercile": ["below", "normal", "above"], "lat": lat, "lon": lon})
    fig = plot_tercile_forecast(pr)   # no style -> legacy path
    assert fig is not None
    plt.close(fig)


def test_plot_terciles_styled_single_bin_legend():
    """A valid single-bin TercileStyle (n=1) must not IndexError in the legend's
    'weak' swatch, which used to index colors[1] unconditionally."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast
    from deepscale.plotting import TercileStyle
    lat = np.linspace(-5, 5, 6); lon = np.linspace(30, 45, 8)
    probs = np.zeros((3, 6, 8)); probs[2] = 0.6; probs[1] = 0.25; probs[0] = 0.15
    pr = xr.DataArray(probs, dims=["tercile", "lat", "lon"],
                      coords={"tercile": ["below", "normal", "above"], "lat": lat, "lon": lon})
    style = TercileStyle(below_colors=["#f8d808"], normal_colors=["#eefcff"],
                         above_colors=["#38f838"], prob_bins=[33.3, 100.01])
    fig = plot_tercile_forecast(pr, style=style, legend=True)
    assert fig is not None
    plt.close(fig)


def test_plot_deterministic_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_deterministic_forecast

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(2).randn(n_lat, n_lon),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_deterministic_forecast(da, title="test")
    assert fig is not None
    plt.close(fig)


def test_plot_exceedance_probability_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_exceedance_probability

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(3).uniform(0, 1, (n_lat, n_lon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_exceedance_probability(da, threshold=100.0)
    assert fig is not None
    plt.close(fig)


def test_plot_flex_pdf_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_flex_pdf

    fig = plot_flex_pdf(
        fcst_mu=2.5, fcst_scale=1.2,
        climo_mu=2.0, climo_scale=1.5,
        location=(35.0, 0.0),
    )
    assert fig is not None
    plt.close(fig)


# ===================================================================
# Reliability diagram (paired with metrics/reliability tests in test_metrics)
# ===================================================================

def test_plot_reliability_diagram_smoke(synthetic_obs):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.reliability import plot_reliability_diagram

    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    fig = plot_reliability_diagram(forecast, synthetic_obs)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# 18b. EOF / CCA mode plots (§3.2)
# ===================================================================

def _build_dual_grid_fixture(seed=0, n_years=25, signal_amp=2.0, noise_amp=0.3):
    """Synthetic SST→precip dual-grid fixture (duplicated from test_methods.py).

    Predictor: 'tropical Pacific' SST on a coarse 6x8 grid (lat ±10°, lon 180-240°).
    Predictand: 'East Africa' precip on a fine 12x12 grid (lat -5 to 15°, lon 30-50°).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(3)

    p_lat = np.linspace(-10, 10, 6)
    p_lon = np.linspace(180, 240, 8)
    o_lat = np.linspace(-5, 15, 12)
    o_lon = np.linspace(30, 50, 12)

    t = np.arange(n_years)
    time_signal = np.sin(2 * np.pi * t / 5.0)

    p_pattern = np.outer(np.sin(np.deg2rad(p_lat) * 3), np.cos(np.deg2rad(p_lon) * 2))
    o_pattern = np.outer(np.cos(np.deg2rad(o_lat) * 2), np.sin(np.deg2rad(o_lon) * 4))

    p_signal = signal_amp * time_signal[:, None, None] * p_pattern[None, :, :]
    o_signal = signal_amp * time_signal[:, None, None] * o_pattern[None, :, :]

    p_noise = rng.standard_normal((n_years, len(members), len(p_lat), len(p_lon))) * noise_amp
    o_noise = rng.standard_normal((n_years, len(o_lat), len(o_lon))) * noise_amp

    predictor = xr.DataArray(
        p_signal[:, None, :, :] + p_noise + 290.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": p_lat, "lon": p_lon},
    )
    predictand = xr.DataArray(
        o_signal + o_noise + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": o_lat, "lon": o_lon},
    )
    return predictor, predictand, o_pattern


def test_apply_sign_convention_flips_negative_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[-3.0, 1.0], [0.5, -0.5]])
    flipped, sign = _apply_sign_convention(arr)
    assert sign == -1.0
    np.testing.assert_array_equal(flipped, -arr)
    # After flip, the dominant lobe is positive.
    assert flipped.flat[int(np.nanargmax(np.abs(flipped)))] > 0


def test_apply_sign_convention_keeps_positive_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[3.0, -1.0], [0.5, -0.5]])
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    np.testing.assert_array_equal(out, arr)


def test_apply_sign_convention_handles_all_nan():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.full((2, 2), np.nan)
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    assert np.all(np.isnan(out))


def _fit_cca_for_mode_plots():
    """Helper: fit CCAMethod on the dual-grid fixture for plotting tests."""
    from deepscale.methods.cca import CCAMethod
    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)
    return m, predictor, predictand


def test_plot_eof_modes_predictor_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=3)
    assert fig is not None
    # 3 mode panels (plus colorbars are extra axes)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 3
    plt.close(fig)


def test_plot_eof_modes_predictand_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictand", n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2
    plt.close(fig)


def test_plot_eof_modes_invalid_kind_raises():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    with pytest.raises(ValueError, match="kind"):
        plot_eof_modes(m, kind="bogus")


def test_plot_eof_modes_caps_n_modes_at_available():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    # Ask for more modes than were fitted; should silently cap.
    fig = plot_eof_modes(m, kind="predictor", n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == m.eofx_.shape[1]
    plt.close(fig)


def test_plot_eof_modes_title_includes_variance_fraction():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=2)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert all("EOF" in t for t in titles)
    assert all("var" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_returns_paired_grid():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    # 2 modes x (predictor + predictand) = 4 map panels
    assert len(map_axes) == 4
    plt.close(fig)


def test_plot_cca_modes_title_includes_canonical_correlation():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=1)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert any("predictor" in t for t in titles)
    assert any("predictand" in t for t in titles)
    assert all("r=" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_caps_at_available_modes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2 * m.ncc_
    plt.close(fig)


def test_mode_plots_dual_grid_integration(tmp_path):
    """Integration: fit CCA on the dual-grid fixture and render both mode plots to disk."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.methods.cca import CCAMethod
    from deepscale.plotting.modes import plot_eof_modes, plot_cca_modes

    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)

    eof_path = tmp_path / "eof_predictor.png"
    cca_path = tmp_path / "cca_modes.png"
    fig_eof = plot_eof_modes(m, kind="predictor", n_modes=3)
    fig_eof.savefig(eof_path, dpi=80)
    plt.close(fig_eof)
    fig_cca = plot_cca_modes(m, n_modes=2)
    fig_cca.savefig(cca_path, dpi=80)
    plt.close(fig_cca)

    # Both files exist and are non-trivially sized (a blank figure is much smaller).
    assert eof_path.exists() and eof_path.stat().st_size > 5000
    assert cca_path.exists() and cca_path.stat().st_size > 5000


def test_plot_cca_modes_pair_shares_sign_convention():
    """Predictor and predictand of a CCA pair should be flipped together."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import (
        plot_cca_modes, _apply_sign_convention, _reconstruct_spatial,
    )
    import matplotlib.pyplot as plt
    m, _, _ = _fit_cca_for_mode_plots()

    # Manually compute what the locked-sign predictor / predictand patterns should be
    # for mode 0, then check the rendered colour-meshes' raw arrays match.
    p_raw = _reconstruct_spatial(
        (m.eofx_ @ m.s_.T)[:, 0], m.x_valid_, m.predictor_shape_
    )
    o_raw = _reconstruct_spatial(
        (m.eofy_ @ m.r_)[:, 0], m.y_valid_, m.predictand_shape_
    )
    p_signed, sign = _apply_sign_convention(p_raw)
    o_signed = o_raw * sign

    fig = plot_cca_modes(m, n_modes=1)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    p_mesh = map_axes[0].collections[0].get_array().reshape(m.predictor_shape_)
    o_mesh = map_axes[1].collections[0].get_array().reshape(m.predictand_shape_)
    np.testing.assert_allclose(np.asarray(p_mesh), p_signed, equal_nan=True)
    np.testing.assert_allclose(np.asarray(o_mesh), o_signed, equal_nan=True)
    plt.close(fig)


def test_somalia_territory_merge_and_border_suppression():
    """clip_to=['Somalia'] must include Somaliland (UN/GHACOF depiction), and the
    internal de-facto line must be suppressed from drawn borders while shared
    exterior borders (Ethiopia-Somalia) survive."""
    from shapely.geometry import Point, box
    from deepscale.plotting.forecasts import (
        _country_geometry, _neutral_border_geoms, _is_internal_border)
    import cartopy.io.shapereader as shpreader

    # Hargeisa (Somaliland capital) is inside the 'Somalia' clip geometry.
    assert _country_geometry(["Somalia"]).contains(Point(44.06, 9.56))

    rdr = shpreader.Reader(shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_boundary_lines_land"))
    records = list(rdr.records())
    kept = _neutral_border_geoms("50m")
    suppressed = [r for r in records if _is_internal_border(r.geometry)]
    # a small number of internal lines suppressed, incl. the Somaliland split
    assert 1 <= len(suppressed) <= 3
    assert len(kept) == len(records) - len(suppressed)
    horn = box(47.5, 7.5, 49.5, 11.8)
    assert any(s.geometry.intersects(horn) for s in suppressed), \
        "the Somalia-Somaliland line must be among the suppressed"
    # the Ethiopia-Somalia border (FEATURECLA 'Indefinite', south of the split)
    # must be KEPT: at least one kept line crosses its bbox interior.
    eth_som = box(43.0, 4.5, 47.5, 7.9)
    assert any(g.intersects(eth_som) for g in kept), \
        "Ethiopia-Somalia border must survive the filter"
