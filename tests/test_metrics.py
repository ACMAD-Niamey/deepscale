import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 4. RPSS metric
# ===================================================================

def test_rpss_perfect_forecast(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    score = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert score > 0.9  # near-perfect


def test_rpss_climatology_forecast(climatology_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    score = m.compute(climatology_forecast, synthetic_obs)
    np.testing.assert_allclose(score, 0.0, atol=0.05)


def test_rpss_worse_than_climatology(synthetic_obs):
    """Inverted forecast should give negative RPSS."""
    from deepscale.metrics.rpss import RPSSMetric
    years = synthetic_obs.year.values
    fine_lat = synthetic_obs.lat.values
    fine_lon = synthetic_obs.lon.values

    from deepscale.metrics.rpss import _cpt_boundaries
    t33, t67 = _cpt_boundaries(synthetic_obs.values)
    t33_da = xr.DataArray(t33, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]})
    t67_da = xr.DataArray(t67, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]})
    cat = xr.where(t33_da > synthetic_obs, 0, xr.where(t67_da > synthetic_obs, 1, 2))

    # Invert: assign high probability to wrong category
    inverted = xr.concat([(cat == (2 - i)).astype(float) for i in range(3)], dim="tercile")
    inverted["tercile"] = [0, 1, 2]

    m = RPSSMetric()
    score = m.compute(inverted, synthetic_obs)
    assert score < 0


def test_rpss_shape_spatial(climatology_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    result = m.compute(climatology_forecast, synthetic_obs, spatial=True)
    assert isinstance(result, xr.DataArray)
    assert "lat" in result.dims and "lon" in result.dims


def test_cpt_boundaries_masks_degenerate_cells():
    """t33 == t67 means the middle third of the sample is a single value, so
    there's no usable tercile partition for that cell. Mask it NaN instead of
    silently bucketing every obs into tercile 2 via the strict `>` rule.

    Regression: the 2026-05-18 nightly run produced NaN for rpss/reliability/
    groc on kenya + nigeria because dry-season CHIRPS cells (>= 2/3 zeros)
    yielded t33 = t67 = 0, and the categorization collapsed silently.
    """
    from deepscale.metrics.rpss import _cpt_boundaries

    n_year = 24
    # Mix of three cell types in a tiny (2, 2) spatial grid:
    #   (0,0) all-zero — degenerate
    #   (0,1) mostly-zero with a few drizzle years — also degenerate (t33==t67==0)
    #   (1,0) constant nonzero — degenerate (t33==t67==const)
    #   (1,1) varied — non-degenerate
    rng = np.random.default_rng(0)
    obs = np.zeros((n_year, 2, 2))
    obs[rng.choice(n_year, 4, replace=False), 0, 1] = rng.uniform(0.5, 5, 4)
    obs[:, 1, 0] = 7.0
    obs[:, 1, 1] = rng.gamma(2.0, 50.0, size=n_year)

    t33, t67 = _cpt_boundaries(obs)
    assert np.isnan(t33[0, 0]) and np.isnan(t67[0, 0])
    assert np.isnan(t33[0, 1]) and np.isnan(t67[0, 1])
    assert np.isnan(t33[1, 0]) and np.isnan(t67[1, 0])
    assert np.isfinite(t33[1, 1]) and np.isfinite(t67[1, 1])
    assert t33[1, 1] < t67[1, 1]


def test_rpss_survives_mixed_degenerate_and_varied_cells(synthetic_obs):
    """A domain with some all-zero (dry) cells alongside normal cells should
    yield a real RPSS computed over the non-degenerate cells, not NaN across
    the board. Companion to the 2026-05-18 nightly regression — before the
    `_cpt_boundaries` mask, even one degenerate cell could be enough to ruin
    the pooled score; after the mask, the metric averages only valid cells.
    """
    from deepscale.metrics.rpss import RPSSMetric, _cpt_boundaries

    # Force the first lat slice to all-zero (a dry-season strip), leaving the
    # rest of the synthetic obs intact.
    obs = synthetic_obs.copy()
    obs.values[:, 0, :] = 0.0

    t33, t67 = _cpt_boundaries(obs.values)
    # The all-zero strip must be masked, the rest must remain finite.
    assert np.all(np.isnan(t33[0, :]))
    assert np.any(np.isfinite(t33[1:, :]))

    # A climatological 1/3,1/3,1/3 forecast yields RPSS ≈ 0 on the valid cells
    # (and the dry strip is NaN-masked out of the average), so the pooled
    # score must be finite — NOT NaN.
    n_year = obs.sizes["year"]
    n_lat = obs.sizes["lat"]
    n_lon = obs.sizes["lon"]
    fcst_vals = np.ones((n_year, 3, n_lat, n_lon)) / 3.0
    forecast = xr.DataArray(
        fcst_vals,
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": obs.year, "tercile": [0, 1, 2],
            "lat": obs.lat, "lon": obs.lon,
        },
    )
    score = RPSSMetric().compute(forecast, obs)
    assert np.isfinite(score), f"expected finite RPSS, got {score}"


# ===================================================================
# 5. ROC metric
# ===================================================================

def test_roc_perfect_discrimination(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert "roc_bn" in result
    assert "roc_an" in result
    assert result["roc_bn"] > 0.8  # should be near 1.0


def test_roc_no_discrimination(climatology_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(climatology_forecast, synthetic_obs)
    # Uniform forecast => no discrimination => ROC ~0.5
    np.testing.assert_allclose(result["roc_bn"], 0.5, atol=0.15)


def test_roc_per_tercile(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert "roc_bn" in result
    assert "roc_nn" in result
    assert "roc_an" in result


# ===================================================================
# 6. Pearson metric
# ===================================================================

def test_pearson_perfect(synthetic_obs):
    from deepscale.metrics.pearson import PearsonMetric
    m = PearsonMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_pearson_zero():
    from deepscale.metrics.pearson import PearsonMetric
    np.random.seed(123)
    years = np.arange(100)
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    b = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    m = PearsonMetric()
    score = m.compute(a, b)
    assert abs(score) < 0.2


# ===================================================================
# 6b. RMSE metric
# ===================================================================

def test_rmse_perfect(synthetic_obs):
    from deepscale.metrics.rmse import RMSEMetric
    m = RMSEMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 0.0, atol=1e-10)


def test_rmse_constant_mean(synthetic_obs):
    from deepscale.metrics.rmse import RMSEMetric
    # Forecast = climatological mean broadcast back across years.
    # RMSE per grid cell should equal the population std (ddof=0) per grid cell.
    forecast = synthetic_obs.mean("year") + 0 * synthetic_obs  # broadcast trick
    m = RMSEMetric()
    spatial_rmse = m.compute(forecast, synthetic_obs, spatial=True)
    expected = synthetic_obs.std("year")  # xarray default is ddof=0
    np.testing.assert_allclose(spatial_rmse.values, expected.values, atol=1e-10)


def test_rmse_alias_registered():
    from deepscale.registry import get_metric
    from deepscale.metrics.rmse import RMSEMetric
    assert get_metric("root_mean_squared_error") is RMSEMetric
    assert get_metric("rmse") is RMSEMetric


# ===================================================================
# 6c. HSS metric
# ===================================================================

def test_hss_perfect(synthetic_obs):
    from deepscale.metrics.heidke import HSSMetric
    from deepscale.metrics.rpss import _cpt_boundaries

    t33, t67 = _cpt_boundaries(synthetic_obs.values)
    obs_vals = synthetic_obs.values
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    n_year, n_lat, n_lon = obs_vals.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for c in range(3):
        fcst[:, c, :, :] = (obs_cat == c).astype(float)

    forecast = xr.DataArray(
        fcst,
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )

    m = HSSMetric()
    score = m.compute(forecast, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=1e-10)


def test_hss_no_skill(synthetic_obs):
    from deepscale.metrics.heidke import HSSMetric

    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    fcst[:, 1, :, :] = 1.0  # always pick "normal" (middle tercile)

    forecast = xr.DataArray(
        fcst,
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )

    m = HSSMetric()
    score = m.compute(forecast, synthetic_obs)
    assert abs(score) < 0.1


def test_hss_alias_registered():
    from deepscale.registry import get_metric
    from deepscale.metrics.heidke import HSSMetric
    assert get_metric("heidke_skill_score") is HSSMetric
    assert get_metric("hss") is HSSMetric


# ===================================================================
# 6d. Spearman metric
# ===================================================================

def test_spearman_perfect(synthetic_obs):
    from deepscale.metrics.spearman import SpearmanMetric
    m = SpearmanMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_spearman_monotonic_nonlinear():
    from deepscale.metrics.spearman import SpearmanMetric
    from deepscale.metrics.pearson import PearsonMetric

    # Centered Gaussian data: x ~ N(0, 1). Cubing strongly distorts linearity
    # while preserving rank order. Theoretical Pearson(X, X^3) for X ~ N(0,1)
    # is 3/sqrt(15) ≈ 0.775; gap from Spearman = 1.0 is ~0.22.
    np.random.seed(7)
    n_year, n_lat, n_lon = 100, 5, 5
    x = np.random.randn(n_year, n_lat, n_lon)
    obs = xr.DataArray(
        x,
        dims=["year", "lat", "lon"],
        coords={
            "year": np.arange(n_year),
            "lat": np.linspace(-1, 1, n_lat),
            "lon": np.linspace(0, 1, n_lon),
        },
    )
    forecast = obs ** 3  # monotonic but strongly nonlinear over [-3, 3]

    spearman = SpearmanMetric().compute(forecast, obs)
    pearson = PearsonMetric().compute(forecast, obs)

    assert spearman > 0.999, f"expected spearman ≈ 1, got {spearman}"
    assert pearson < spearman - 0.01, (
        f"expected pearson strictly below spearman by >= 0.01; "
        f"got pearson={pearson}, spearman={spearman}"
    )


def test_spearman_zero():
    from deepscale.metrics.spearman import SpearmanMetric
    np.random.seed(123)
    years = np.arange(100)
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    b = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    m = SpearmanMetric()
    score = m.compute(a, b)
    assert abs(score) < 0.2


# ===================================================================
# 16. 2AFC metric
# ===================================================================

def test_2afc_perfect(synthetic_obs):
    from deepscale.metrics.two_afc import TwoAFCMetric
    score = TwoAFCMetric().compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_2afc_uniform_random():
    from deepscale.metrics.two_afc import TwoAFCMetric
    np.random.seed(42)
    n_year, n_lat, n_lon = 100, 5, 5
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    forecast = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                            dims=["year", "lat", "lon"], coords=coords)
    score = TwoAFCMetric().compute(forecast, obs)
    assert abs(score - 0.5) < 0.05


def test_2afc_constant_forecast_no_skill(synthetic_obs):
    from deepscale.metrics.two_afc import TwoAFCMetric
    forecast = synthetic_obs * 0 + 1.0  # all-constant
    score = TwoAFCMetric().compute(forecast, synthetic_obs)
    # Half-credit-for-ties: a constant forecast scores 0.5 (matches the
    # issue's "constant forecast (no skill) ≈ 0.5" criterion).
    np.testing.assert_allclose(score, 0.5, atol=1e-12)


# ===================================================================
# 17. Per-tercile ROC variants
# ===================================================================

def test_roc_area_below_normal_matches_roc_bn(synthetic_obs, perfect_tercile_forecast):
    from deepscale.registry import get_metric
    full = get_metric("roc")().compute(perfect_tercile_forecast, synthetic_obs)
    bn = get_metric("roc_area_below_normal")().compute(perfect_tercile_forecast, synthetic_obs)
    np.testing.assert_allclose(bn, full["roc_bn"], atol=1e-12)


def test_roc_area_above_normal_matches_roc_an(synthetic_obs, perfect_tercile_forecast):
    from deepscale.registry import get_metric
    full = get_metric("roc")().compute(perfect_tercile_forecast, synthetic_obs)
    an = get_metric("roc_area_above_normal")().compute(perfect_tercile_forecast, synthetic_obs)
    np.testing.assert_allclose(an, full["roc_an"], atol=1e-12)


# ===================================================================
# 18. Reliability metric
# (plot_reliability_diagram smoke test lives in test_plotting.py)
# ===================================================================

def test_reliability_climatology(synthetic_obs):
    from deepscale.metrics.reliability import ReliabilityMetric
    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0  # uniform climatology
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    rel = ReliabilityMetric().compute(forecast, synthetic_obs)
    assert rel < 0.05, f"expected near-perfect calibration, got {rel}"


def test_reliability_overconfident(synthetic_obs):
    from deepscale.metrics.reliability import ReliabilityMetric
    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    fcst[:, 0, :, :] = 1.0  # always confident BN
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    rel = ReliabilityMetric().compute(forecast, synthetic_obs)
    assert rel > 0.2, f"expected badly calibrated forecast, got {rel}"


# ===================================================================
# 19. Metric presets (#52)
# ===================================================================

def test_skill_preset_svslrf(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="svslrf")
    assert "rpss" in report.scores
    assert "roc_bn" in report.scores  # from "roc" metric (returns dict)
    assert "roc_nn" in report.scores
    assert "roc_an" in report.scores
    assert "reliability" in report.scores


def test_skill_preset_all_dedupes_aliases(synthetic_obs, perfect_tercile_forecast):
    import warnings
    import deepscale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="all")
    # Alias dedup: regardless of whether a given metric computes on this
    # input, the registry must never emit two keys for the same class. RMSE
    # in particular now raises on a tercile-probability forecast, so 0 keys
    # is the actual outcome here — the dedup contract is "at most one".
    rmse_keys = [k for k in report.scores if k in ("rmse", "root_mean_squared_error")]
    assert len(rmse_keys) <= 1, f"alias dedup broken; got {rmse_keys}"
    hss_keys = [k for k in report.scores if k in ("hss", "heidke_skill_score")]
    assert len(hss_keys) <= 1, f"alias dedup broken; got {hss_keys}"

    # spread_error_* require a `member` dim; the tercile forecast doesn't
    # have one, so metrics="all" should skip them with a warning rather than
    # abort the whole report.
    assert "spread_error_ratio" not in report.scores
    assert "spread_error_correlation" not in report.scores
    skipped = [str(w.message) for w in caught if "spread_error" in str(w.message)]
    assert skipped, "expected a skip-warning for spread_error_* metrics"


def test_skill_bare_string_single_metric(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="rpss")
    assert "rpss" in report.scores


def test_skill_list_metrics_still_works(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    # Two probabilistic metrics that are valid on a tercile-probability forecast.
    # The point of this test is that the explicit-list path still works after
    # the metrics="all" feature was added.
    report = deepscale.skill(
        perfect_tercile_forecast, synthetic_obs,
        metrics=["rpss", "heidke_skill_score"],
    )
    assert set(report.scores.keys()) >= {"rpss", "heidke_skill_score"}


# ===================================================================
# 19. Spread-error metric
# ===================================================================

def test_spread_error_ratio_calibrated():
    """Calibrated ensemble: per-year ensemble-mean bias matched to per-member
    spread so spread ≈ error.

    Construction: each member = truth + bias_y(lat, lon) + member_noise. The
    bias is shared across members (so it contributes to error but not spread).
    The member noise contributes to spread (std σ) and adds a small σ/√N
    component to error. With bias std B = σ·√(π/2), E[|bias|] = σ, so
    mean(spread) ≈ mean(error) ≈ σ.
    """
    from deepscale.metrics.spread_error import SpreadErrorRatioMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 200, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    sigma = 1.0
    B = sigma * np.sqrt(np.pi / 2)
    bias = np.random.randn(n_year, n_lat, n_lon) * B  # shared across members
    member_noise = np.random.randn(n_year, n_member, n_lat, n_lon) * sigma
    fcst = truth[:, None, :, :] + bias[:, None, :, :] + member_noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs)
    assert 0.85 < ratio < 1.15, f"expected ~1, got {ratio}"


def test_spread_error_ratio_underdispersed():
    """Spread = 0.1 × error → ratio ≈ 0.1."""
    from deepscale.metrics.spread_error import SpreadErrorRatioMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 30, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    # Add a large per-year bias so |mean - obs| is dominated by the bias,
    # and shrink the member-axis noise so spread is 10× smaller.
    bias = np.random.randn(n_year, 1, n_lat, n_lon) * 5.0
    small_noise = np.random.randn(n_year, n_member, n_lat, n_lon) * 0.5
    fcst = truth[:, None, :, :] + bias + small_noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs)
    assert ratio < 0.3, f"expected strongly underdispersed (<0.3), got {ratio}"


def test_spread_error_correlation_tracks():
    """Ensemble where high-spread years are also high-error years.

    Construct an ensemble whose per-year noise amplitude varies with year;
    the ensemble mean's distance from truth grows with that amplitude, so
    spread and error track each other strongly.
    """
    from deepscale.metrics.spread_error import SpreadErrorCorrelationMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 30, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    amplitude = np.linspace(0.2, 4.0, n_year)  # year-varying noise level
    noise = (
        np.random.randn(n_year, n_member, n_lat, n_lon)
        * amplitude[:, None, None, None]
    )
    fcst = truth[:, None, :, :] + noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    r = SpreadErrorCorrelationMetric().compute(forecast, obs)
    assert r > 0.7, f"expected strong positive spread-error correlation, got {r}"


def test_spread_error_no_member_raises():
    """Forecast without a 'member' dim is a usage error."""
    from deepscale.metrics.spread_error import (
        SpreadErrorRatioMetric,
        SpreadErrorCorrelationMetric,
    )

    n_year, n_lat, n_lon = 10, 4, 4
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )

    for cls in (SpreadErrorRatioMetric, SpreadErrorCorrelationMetric):
        with pytest.raises(ValueError, match="member"):
            cls().compute(forecast, obs)


def test_spread_error_spatial_returns_dataarray():
    """spatial=True collapses only the year dim and returns a DataArray."""
    from deepscale.metrics.spread_error import (
        SpreadErrorRatioMetric,
        SpreadErrorCorrelationMetric,
    )

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 10, 4, 5, 6
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_member, n_lat, n_lon),
        dims=["year", "member", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs, spatial=True)
    corr = SpreadErrorCorrelationMetric().compute(forecast, obs, spatial=True)

    for result in (ratio, corr):
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"lat", "lon"}
        assert result.sizes == {"lat": n_lat, "lon": n_lon}


def test_spread_error_diagnostics_pairs():
    """Helper returns per-year spread and error series of equal length."""
    from deepscale.metrics.spread_error import (
        SpreadErrorDiagnostics,
        spread_error_diagnostics,
    )

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 8, 5, 3, 3
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_member, n_lat, n_lon),
        dims=["year", "member", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    diag = spread_error_diagnostics(forecast, obs)
    assert isinstance(diag, SpreadErrorDiagnostics)
    assert diag.spread.dims == ("year",)
    assert diag.error.dims == ("year",)
    assert diag.spread.sizes["year"] == n_year
    assert diag.error.sizes["year"] == n_year

    diag_sp = spread_error_diagnostics(forecast, obs, spatial=True)
    assert set(diag_sp.spread.dims) == {"year", "lat", "lon"}
    assert set(diag_sp.error.dims) == {"year", "lat", "lon"}


# ===================================================================
# 20. Generalized ROC (GROC)
# ===================================================================

def test_groc_perfect_forecast(synthetic_obs):
    """A forecast that puts all probability mass on the correct tercile gives GROC = 1.0."""
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric
    from deepscale.metrics.rpss import _cpt_boundaries

    obs_vals = synthetic_obs.values
    t33, t67 = _cpt_boundaries(obs_vals)
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    n_year, n_lat, n_lon = obs_vals.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for k in range(3):
        fcst[:, k, :, :] = (obs_cat == k).astype(float)

    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    score = GeneralizedROCMetric().compute(forecast, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=1e-9)


def test_groc_climatology_forecast(synthetic_obs):
    """A uniform climatological forecast (1/3, 1/3, 1/3) gives GROC = 0.5."""
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric

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
    score = GeneralizedROCMetric().compute(forecast, synthetic_obs)
    np.testing.assert_allclose(score, 0.5, atol=1e-9)


def test_groc_spatial_returns_dataarray(synthetic_obs):
    """spatial=True collapses year only and returns a (lat, lon) DataArray."""
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric
    from deepscale.metrics.rpss import _cpt_boundaries

    obs_vals = synthetic_obs.values
    t33, t67 = _cpt_boundaries(obs_vals)
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    n_year, n_lat, n_lon = obs_vals.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for k in range(3):
        fcst[:, k, :, :] = (obs_cat == k).astype(float)
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    result = GeneralizedROCMetric().compute(forecast, synthetic_obs, spatial=True)
    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"lat", "lon"}
    assert result.sizes == {"lat": n_lat, "lon": n_lon}
    # Perfect forecast in every cell → each cell should be 1.0.
    np.testing.assert_allclose(result.values, 1.0, atol=1e-9)


def test_groc_single_category_returns_nan():
    """If every obs sample lands in the same tercile, GROC is undefined → NaN + warning."""
    import warnings as _warnings
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric

    # All-constant obs → _cpt_boundaries collapses t33 = t67 = the constant,
    # so every cell falls through both `>` comparisons and lands in the same
    # single category. With <2 distinct labels, GROC is undefined.
    n_year, n_lat, n_lon = 10, 4, 4
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    obs = xr.DataArray(
        np.ones((n_year, n_lat, n_lon)),
        dims=["year", "lat", "lon"], coords=coords,
    )
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={**coords, "tercile": [0, 1, 2]},
    )

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        score = GeneralizedROCMetric().compute(forecast, obs)
    assert np.isnan(score), f"expected NaN, got {score}"
    msgs = [str(w.message) for w in caught if "generalized_roc" in str(w.message)]
    assert msgs, "expected a RuntimeWarning naming the metric"


def test_groc_missing_tercile_raises(synthetic_obs):
    """A forecast without a size-3 'tercile' dim is a usage error."""
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric

    n_year, n_lat, n_lon = synthetic_obs.shape
    forecast = xr.DataArray(  # no 'tercile' dim at all
        np.zeros((n_year, n_lat, n_lon)),
        dims=["year", "lat", "lon"],
        coords={"year": synthetic_obs.year, "lat": synthetic_obs.lat, "lon": synthetic_obs.lon},
    )
    with pytest.raises(ValueError, match="tercile"):
        GeneralizedROCMetric().compute(forecast, synthetic_obs)

    # Wrong-sized tercile dim (size 2 instead of 3) also raises.
    forecast2 = xr.DataArray(
        np.zeros((n_year, 2, n_lat, n_lon)),
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    with pytest.raises(ValueError, match="tercile"):
        GeneralizedROCMetric().compute(forecast2, synthetic_obs)


def test_groc_loo_boundaries_perfect(synthetic_obs):
    """LOO path: build the perfect forecast against LOO-derived categories
    and assert score == 1.0. Confirms the LOO branch is actually used (a
    non-LOO-built perfect forecast would *not* score 1.0 here)."""
    from deepscale.metrics.generalized_roc import (
        GeneralizedROCMetric,
        _obs_to_categories,
    )

    obs_vals = synthetic_obs.values
    obs_cat = _obs_to_categories(obs_vals, loo_boundaries=True)
    n_year, n_lat, n_lon = obs_vals.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for k in range(3):
        fcst[:, k, :, :] = (obs_cat == k).astype(float)
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    score = GeneralizedROCMetric().compute(forecast, synthetic_obs, loo_boundaries=True)
    np.testing.assert_allclose(score, 1.0, atol=1e-9)


def test_groc_pairs_correctly_when_forecast_dims_permuted(synthetic_obs):
    """Permuting forecast's non-tercile dims must not change the score —
    catches the obs/forecast flat-pairing hazard."""
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric
    from deepscale.metrics.rpss import _cpt_boundaries

    obs_vals = synthetic_obs.values
    t33, t67 = _cpt_boundaries(obs_vals)
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))
    n_year, n_lat, n_lon = obs_vals.shape

    # Build a forecast that is NOT perfect — flip 1/4 of the years' labels
    # so the score is sensitive to mispairing rather than collapsing to 1.0.
    rng = np.random.default_rng(0)
    flip = rng.random((n_year, n_lat, n_lon)) < 0.25
    labels = np.where(flip, (obs_cat + 1) % 3, obs_cat)
    fcst_canon = np.zeros((n_year, 3, n_lat, n_lon))
    for k in range(3):
        fcst_canon[:, k, :, :] = (labels == k).astype(float)

    coords = {
        "year": synthetic_obs.year,
        "tercile": [0, 1, 2],
        "lat": synthetic_obs.lat,
        "lon": synthetic_obs.lon,
    }
    forecast_canon = xr.DataArray(
        fcst_canon, dims=["year", "tercile", "lat", "lon"], coords=coords,
    )
    forecast_permuted = forecast_canon.transpose("year", "tercile", "lon", "lat")
    assert forecast_canon.dims != forecast_permuted.dims

    s_canon = GeneralizedROCMetric().compute(forecast_canon, synthetic_obs)
    s_perm = GeneralizedROCMetric().compute(forecast_permuted, synthetic_obs)
    np.testing.assert_allclose(s_perm, s_canon, atol=1e-12)


def test_groc_independent_oracle():
    """Hand-built fixture with obvious terciles (no reuse of _cpt_boundaries
    in the test): obs is the year index repeated per cell, so terciles are
    just the lowest-third, middle-third, highest-third of years.
    """
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric

    n_year, n_lat, n_lon = 12, 2, 2  # exactly divisible into thirds
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    obs_1d = np.arange(n_year, dtype=float)
    obs = xr.DataArray(
        np.broadcast_to(obs_1d[:, None, None], (n_year, n_lat, n_lon)).copy(),
        dims=["year", "lat", "lon"], coords=coords,
    )
    # By hand: years 0-3 → BN, 4-7 → NN, 8-11 → AN.
    expected_label = np.repeat([0, 1, 2], 4)
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for y in range(n_year):
        fcst[y, expected_label[y], :, :] = 1.0
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={**coords, "tercile": [0, 1, 2]},
    )
    score = GeneralizedROCMetric().compute(forecast, obs)
    np.testing.assert_allclose(score, 1.0, atol=1e-9)


def test_groc_alias_registered():
    """Both 'generalized_roc' and 'groc' resolve to the same class."""
    from deepscale.registry import get_metric
    from deepscale.metrics.generalized_roc import GeneralizedROCMetric

    assert get_metric("generalized_roc") is GeneralizedROCMetric
    assert get_metric("groc") is GeneralizedROCMetric


def test_groc_two_category_cell_does_not_crash():
    """A cell whose obs record spans only two tercile categories must yield a
    finite Hand-&-Till pair AUC, not an sklearn multiclass ValueError (seen
    live on dry-masked seasonal domains with spatial=True)."""
    import numpy as np
    from deepscale.metrics.generalized_roc import _groc_from_flat

    rng = np.random.default_rng(7)
    n = 24
    y_true = np.array([0, 2] * (n // 2))              # only below/above present
    y_score = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    # make the forecast informative so the AUC is meaningfully > 0.5
    y_score[y_true == 2, 2] += 1.0
    y_score[y_true == 0, 0] += 1.0
    y_score /= y_score.sum(axis=1, keepdims=True)
    score = _groc_from_flat(y_true, y_score)
    assert np.isfinite(score)
    assert 0.5 < score <= 1.0


def test_groc_single_category_cell_returns_nan():
    import numpy as np
    from deepscale.metrics.generalized_roc import _groc_from_flat

    y_true = np.zeros(10, dtype=int)
    y_score = np.full((10, 3), 1 / 3)
    assert np.isnan(_groc_from_flat(y_true, y_score))
