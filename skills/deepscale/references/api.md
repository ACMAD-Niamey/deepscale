# DeepScale API reference

Import name: `deepscale` (distribution: `accord-deepscale`). `__all__`: `downscale, train, optimize, ensemble, pool_ensembles, skill, SkillReport, skill_compare, ComparisonReport, prediction_error_variance, flex_forecast, FlexForecastResult, seasonal_mme, SeasonalMMEResult, Index, calibrate, LogitConfig, accumulate, frequency_below, percentile_of, rank_of_record, seasonal_reduce, seasonal_stack, AnalogSet, analogs_from_years, analogs_from_index, analogs_from_field, analogs_where, complete, CompletionResult, quantile_map, error_bounds, ErrorBounds, seasonal_coefficients, write_terciles, tercile_mae, combine_terciles, mask_by_skill, dry_mask, loo_predict, loo_corr, permutation_test, fdr, plot_terciles, plot_accumulation_scenarios, plot_index_scatter, plot_field_map, plot_choropleth, natural_earth_borders, TercileStyle, plot_field, render_styled_terciles, plot_tercile_comparison`. Importing the package registers all methods/metrics/strategies.

The analog-selection / scenario-completion / climate-positioning verbs (`AnalogSet`, `analogs_*`, `complete`/`CompletionResult`, `seasonal_stack`, `seasonal_reduce`, `accumulate`, `percentile_of`, `frequency_below`, `rank_of_record`, `quantile_map`, `error_bounds`) form the SMPG subsystem — full signatures and semantics live in [analog-completion.md](analog-completion.md). `combine_terciles`/`mask_by_skill`/`dry_mask` and `pool_ensembles` are documented in [methods.md](methods.md); the significance metrics (`loo_predict`, `loo_corr`, `permutation_test`, `fdr`) in [metrics-and-terciles.md](metrics-and-terciles.md); all plotting in [plotting-reporting.md](plotting-reporting.md). This file covers the core forecasting verbs and the generalized `Index` below.

## `downscale()`

```python
def downscale(predictor_hindcast=None, obs=None, method="bcsd",
              output_type="continuous", weights_path=None, **kwargs)
```

- `method`: registered method name (see `references/methods.md`).
- `output_type`: `"continuous"` (default) or `"tercile"` (requires `obs` for boundaries).
- `weights_path`: load a checkpoint and run inference only (skips fitting).
- Recognized method-constructor kwargs: `n_modes, x_eof_modes, y_eof_modes, cca_modes, device, n_samples, target_variable, variant`. Other kwargs: `verbose=True`, `forecast=` (explicit forecast field).
- Auto-split: if `predictor_hindcast` has a `year` dim and no explicit `forecast`, fits on `year[:-1]` and predicts the last year (needs ≥2 years).
- `gcm=` is a deprecated alias for `predictor_hindcast=` (DeprecationWarning; both → `TypeError`).
- `requires_training=True` methods raise `RuntimeError` directing you to `train()` + `weights_path=`.
- Probabilistic methods (`ProbabilisticMethodBase`) with `output_type="tercile"` go through `predict_distribution()` + `to_tercile(..., method="counting")`.
- Returns: `(member, lat, lon)` continuous, or `(tercile, lat, lon)` probabilities.

## `optimize()`

```python
def optimize(gcm, obs, methods=None, cv="loyo", primary_metric="rpss", **kwargs)
    -> OptimizeResult   # .method: str, .score: float, .forecast: xr.DataArray (continuous)
```

- `methods` default `["bcsd", "cca"]`. `cv` is a scheme name or callable (e.g. `partial(expanding, min_train=4)`).
- Per method: runs CV folds, converts each held-out prediction with `to_tercile(pred, obs_train)` (train-only obs — no leakage), scores the pooled CV forecast, then refits the winner on all data for `best_forecast`.
- kwargs: `progress=True`, `verbose=True`, plus method-constructor params.
- `primary_metric` must be a scalar-valued (leaf) metric.

## `train()`

```python
def train(method_name, hindcast, obs, save_to=None, *, verbose=True, **kwargs) -> MethodBase
```

Fits a registered method once; `save_to=` checkpoints via `MethodBase.save`. The returned instance has `.predict(gcm_field)`, `.is_trained`, `.save(path)` / `MethodBase.load(path)`. Statistical methods fit in seconds; only `requires_training=True` (DL) methods *require* this path.

## `calibrate()` and `LogitConfig`

```python
def calibrate(predictor=None, obs=None, *, method, forecast=None,
              forecast_year=None, predictor_hindcast=None,
              predictor_forecast=None, combine="mean", verbose=False,
              **method_kwargs) -> xr.DataArray   # (tercile, lat, lon), sums to 1
```

MOS calibration **without changing resolution**. `method` is `"ereg"`, `"logit"`, `"smoothed_regression"`, or a `LogitConfig`. `predictor`/`predictor_hindcast` are aliases, as are `forecast`/`predictor_forecast`. `combine` supports only `"mean"` (per-model tercile maps averaged skipna, then renormalized onto the probability simplex; uncalibratable cells → NaN). The `(tercile, lat, lon)` return shape above is the `ereg`/`logit` case; `smoothed_regression` is season-aware and returns `(season, lat, lon)` (deterministic) or `(season, tercile, lat, lon)` (tercile).

**eReg** — predictor `{model: (hindcast, forecast_or_None)}` or a single `(hindcast, forecast)` tuple, gridded, **already on the obs grid** (mismatch raises `ValueError`). Per grid cell: OLS of obs on ensemble-mean hindcast → Gaussian terciles with leverage-inflated prediction-error variance (Wilks 2006 eq. 6.22). Extra kwargs: `clip_negative=False` (precip), `threshold_source="obs"|"fitted"`, `native_years=False` (calibrate each model on its own year overlap with obs; floor 3 years), `forecast_year` (inferred if every forecast has exactly one year, else defaults to max obs year).

**logit** — predictor `{model: index_series}` or one index series (dims: `year`); `forecast=` matching per-model scalar index value(s). Extra kwargs: `model="independent_binomial"|"multinomial"`, `backend="sklearn"|"statsmodels"` (auto-selects statsmodels when `significance` is set), `regularization=None`, `significance_mask=None`, `min_years=10`, `tercile_edges="exclusive"|"inclusive"`, `detrend=False`.

**smoothed_regression** — Kharin et al. (2017) smoothed-coefficient postprocessing; a single ensemble hindcast (not a dict), season-aware: `predictor` `(season, year, member, lat, lon)`, `obs` `(season, year, lat, lon)`, same grid. Extra kwargs: `output_type="deterministic"|"tercile"` (default deterministic → `(season, lat, lon)`; tercile → `(season, tercile, lat, lon)`), `temporal_sigma=None|float|"constant"` (per-season / cyclic Gaussian across the seasonal cycle / one year-round coefficient), `distribution="normal"|"gamma"` (tercile only; gamma maps members+obs through the gamma CDF for non-negative variables), `constrained=True` (tercile only; analytic spread vs `False` CRPS-minimizing), `forecast_year` (must be a year present in the hindcast). Fit-and-apply on the hindcast: a separate out-of-sample `forecast=` raises `NotImplementedError`; tercile output without a `member` dim raises `ValueError`. Score deterministic output with `msss`, tercile output with `crpss`/`reliability`.

```python
@dataclass(frozen=True)
class LogitConfig:
    index: Index
    model: str = "independent_binomial"
    predictor_level: str = "model_mean"
    detrend: bool = False
    significance: float | None = None
    regularization: float | None = None
    backend: str | None = None
    min_years: int = 10
```

Pass `method=LogitConfig(index=..., ...)` with **gridded SST** predictors: `calibrate` reduces the fields through `config.index` (an `Index`) then runs logit. `LogitConfig` field names may also be passed as `**method_kwargs` overrides.

## `ensemble()` and `EnsembleResult`

```python
def ensemble(forecasts, obs, *, strategy="uniform", optimize_ensemble=False,
             primary_metric="rpss", safeguards=None, cv="loyo", **kwargs) -> EnsembleResult
```

- `forecasts`: list of DataArrays or objects with `.forecast`/`.method` (e.g. `OptimizeResult`).
- `strategy`: `"uniform" | "skill_weighted" | "bma" | "drop_worst"`. Strategy extras via kwargs: `hindcasts=` (bma), `n_drop=` (drop_worst), `scores=` (skill_weighted/drop_worst).
- `safeguards` defaults: `{"nested_cv": True, "shrinkage": 0.5, "min_effective_n": 3, "gate": True}` (unknown keys → `ValueError`). With `optimize_ensemble=True` (requires `obs`): nested-CV weight optimization with shrinkage toward uniform, an effective-N floor, and an acceptance gate that falls back to uniform when optimized CV skill < uniform CV skill (`RuntimeWarning`).

```python
@dataclass
class EnsembleResult:
    forecast: xr.DataArray
    weights: np.ndarray
    member_names: list
    member_cv_skill: dict = {}
    effective_n: float = 0.0
    gate_passed: bool = True
    shrinkage_lambda: float = 0.0
    safeguards_applied: dict = {}
    pev: xr.DataArray | None = None            # per-cell CV prediction error variance
    member_contributions: dict | None = None   # {name: {correlation_with_mme_mean, skill_delta}}
```

`pev`/`member_contributions` populate only when honest CV predictions exist (year-dim forecasts + obs).

## `skill()` and `SkillReport`

```python
def skill(forecast, obs, metrics=None, spatial=False, **kwargs) -> SkillReport
```

- `metrics`: `None` (→ `["rpss"]`), a name, a preset (`"svslrf"` → rpss/roc/reliability; `"all"` → every registered metric, shape-incompatible ones skipped with `RuntimeWarning`), or a list.
- `spatial=True`: per-cell maps in `.spatial`, scalar means in `.scores`.
- Extra `**kwargs` are forwarded to every metric's `compute()` — this is how per-metric options like `loo_boundaries=True`, `cv_window=3`, `bounded=True` (RPSS) or `n_bins` (reliability) are passed through `skill()`.

```python
@dataclass
class SkillReport:
    scores: dict     # metric -> scalar
    spatial: dict    # metric -> (lat, lon) DataArray
    metadata: dict
    diagrams: dict   # ROC curves, reliability bins, member contributions
    def to_table(self) -> pandas.DataFrame        # ['metric', 'value']
    def to_dict(self) -> dict                     # JSON-round-trippable
    def to_geotiff(self, path, metric)            # EPSG:4326 GeoTIFF (rioxarray)
    def to_pdf(self, path, *, style="svslrf")     # WMO-SVSLRF report (plotting extra)
```

## `skill_compare()` and `ComparisonReport`

```python
def skill_compare(forecasts, obs, metrics=None, spatial=False) -> ComparisonReport
```

`forecasts` is `{method_name: forecast}`. **No regridding** — any lat/lon mismatch with obs raises `ValueError`.

```python
@dataclass
class ComparisonReport:
    reports: dict; methods: list; metrics: list
    def to_table(self) -> DataFrame                   # methods x metrics
    def to_heatmap(self, path=None, *, metric=None)   # matplotlib Figure (RdBu, ±1)
    def to_pdf(self, path, *, spatial_maps=False)
```

## `prediction_error_variance()`

```python
def prediction_error_variance(cv_predictions, obs) -> xr.DataArray  # (lat, lon)
```

Year-mean squared residual per cell (no dof correction/leverage). Both inputs must have `year` and cover **the same year set** (`ValueError` otherwise). For deterministic methods.

## `flex_forecast()` and `FlexForecastResult`

```python
def flex_forecast(det_fcst, pev, obs, threshold, is_percentile=True,
                  distribution="gaussian") -> FlexForecastResult
```

Per-cell exceedance probability P(Y > threshold). `det_fcst` `(lat,lon)` = location, `sqrt(pev)` = scale, `obs` `(year,lat,lon)` = climatological reference. `threshold`: climatological quantile in [0,1] (`is_percentile=True`, default) or absolute value. Only `"gaussian"` in V1 (Gamma → `NotImplementedError`). Result fields: `exceedance_prob` (`(lat,lon)` in [0,1]), `fcst_mu`, `fcst_scale`, `climo_mu`, `climo_scale`, `transformed_threshold`, `metadata`, `.to_dict()`.

## `seasonal_mme()` and `SeasonalMMEResult`

```python
def seasonal_mme(predictor_tracks, obs, *, method="cca", cv="loyo",
                 cpt_args=None, skill_metrics=None, tercile_method=None,
                 probability_aggregation="pooled", forecast_year=None,
                 optimize_ensemble=False, primary_metric="rpss",
                 verbose=True, native_years=False) -> SeasonalMMEResult
```

Full PyCPT-style multi-track MME pipeline. `predictor_tracks`: `{track_name: {model_name: (hindcast, forecast_or_None)}}` (e.g. `{"prcp": {...}, "sst": {...}}`). Per-model CV fit/predict → pooled members → uniform ensemble → tercile forecast + CV terciles → skill.

- `probability_aggregation`: `"pooled"` or `"cpt_per_model"` (the latter requires `method="cca"`).
- `tercile_method` default: `"cpt"` for CCA, else `"bootstrap"`.
- `cpt_args` knobs: `n_modes, x_eof_modes, y_eof_modes, cca_modes, standardize, transform_predictand, tailoring, drymask_threshold, crossvalidation_window, mode_selection ("auto"|"cpt"), x_eof_range, y_eof_range, cca_range, mode_selection_fallback, skillmask_threshold`.
- Requires ≥5 intersection years across obs + all hindcasts. `method="corrdiff"` raises `NotImplementedError` (V1 deterministic only).

```python
@dataclass
class SeasonalMMEResult:
    forecast: xr.DataArray            # deterministic MME mean (lat, lon)
    tercile_forecast: xr.DataArray    # (tercile, lat, lon)
    tercile_cv: xr.DataArray          # (year, tercile, lat, lon)
    skill_report: SkillReport
    ensemble_result: EnsembleResult
    pev: xr.DataArray | None
    per_model_methods: dict
    per_model_cv_hindcasts: dict
    per_model_forecasts: dict
    metadata: dict   # years_used, forecast_year, cv, method, tercile_method, n_members, run_at, ...
```

## `Index` (teleconnection / climate indices)

An index is three choices — *which boxes*, *how each box is transformed*, and *how the transformed boxes combine*. Name those three and almost every operational teleconnection index falls out of the same machinery. Nothing here is SST-specific: the boxes are just regions and the field is just a field (e.g. `wpac` runs on precipitation).

```python
@dataclass(frozen=True)
class Index:
    name: str
    regions: Mapping[str, object]
    _combine: Callable[[dict], xr.DataArray]
    transform: str | Mapping[str, str] = "standardize"   # "standardize" | "anomaly" | "raw", or {region: mode}
    weights: object = None                                # None | "cos_lat" | xr.DataArray
    baseline: object = None                               # (start, end) tuple or slice restricting the transform reference

    @classmethod
    def named(cls, name, **overrides) -> Index
    @classmethod
    def custom(cls, *, name, regions, combine,
               transform="standardize", weights=None, baseline=None) -> Index
    @staticmethod
    def list_named() -> dict[str, str]                    # {name: one-line description}
    def reduce(self, field=None, climatology=None, *, baseline=None, sst=None) -> xr.DataArray
```

- **`transform`** — how each box series is transformed before `combine` sees it: `"standardize"` (default, z-score against the reference — the back-compatible behaviour), `"anomaly"` (subtract the reference mean, keeping physical units), or `"raw"` (untouched). A `{region: mode}` mapping sets it per box; regions omitted from a partial mapping default to `"standardize"`. An unknown region name in the mapping raises `ValueError`.
- **`weights`** — `None` (default, plain box mean), `"cos_lat"` (area-weight by cos of latitude — matters for tall regions like RONI's 40°-tall tropical band), or an explicit `xr.DataArray`. The default stays unweighted so the WVG family keeps matching the operational ICPAC / WASS2S references.
- **`baseline`** — restricts the reference used by `standardize`/`anomaly` to a period, e.g. `(1991, 2020)` (WMO) or a `slice`. Ignored by `"raw"`. A `reduce(..., baseline=)` argument overrides the index's own `baseline`.

**`reduce(field, climatology=None, *, baseline=None)`** reduces a gridded field (lat/lon aliases accepted; a `member` dim is averaged out; despite the name it need not be SST) to a 1-D series over the field's time/year dim, or a scalar if it has none. Pass the hindcast as `climatology` when reducing a forecast year so both indices share a scale; it defaults to `field` itself. An **all-`"raw"`** index consults no reference, so it works with no `climatology` even on a single time-less forecast map (e.g. the `>29 °C` `wio` threshold). The first positional argument was renamed `sst → field`; the old `sst=` keyword still works but emits a `DeprecationWarning` (and passing both raises).

**Named indices** (`Index.list_named()` for the live set): `wvg`, `wvg2` (Western-V Gradient, 3-/2-box, standardized+unweighted — ICPAC primary), `nino12`, `nino3`, `nino34`, `nino4` (standardized+unweighted bare Niño boxes), `oni` (Niño3.4 anomaly, °C; anomaly+cos_lat — apply a 3-month running mean upstream for the operational ONI), `roni` (Niño3.4 anomaly minus 20°S–20°N mean, °C; anomaly+cos_lat), `dmi` / alias `iod` (Dipole Mode Index, °C; anomaly+cos_lat), `wtio`, `setio` (IOD pole SST anomalies; anomaly+cos_lat), `wio` (western Indian Ocean SST in absolute units; raw+cos_lat — for thresholds), `wpac` (standardized equatorial West Pacific field mean; also runs on precip as a Walker-circulation indicator). The WVG family and bare Niño indices deliberately keep standardize/unweighted so operational values are unchanged. `Index.named(name, **overrides)` may override `transform`/`weights`/`baseline`, e.g. `Index.named("roni", baseline=(1991, 2020))`.

**`Index.custom(*, name, regions, combine, transform=..., weights=..., baseline=...)`** — `regions` values may each be a bbox list `[lat_s, lat_n, lon_w, lon_e]`, a `{"south":…, "north":…, "west":…, "east":…}` mapping, or the **name of a box in `deepscale.indices.REGIONS`**. Shapefile paths / shapely geometries work only if Rosetta is installed. `combine` receives the transformed regional series keyed by region name, e.g. WVG is `lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3`.

Box definitions live in `deepscale.indices.REGIONS` (a dict of `{south, north, west, east}` in 0–360 lon; `reduce` normalizes whatever convention the field uses). Neither `REGIONS` nor `Index.list_named` is in the top-level `__all__`.

## `seasonal_coefficients()` (Kharin 2017 smoothed regression slopes)

```python
def seasonal_coefficients(predictor_hindcast, obs, temporal_sigma=None) -> xr.DataArray
```

Per-gridpoint ensemble-mean regression slope `a = Cov(Fbar, O)/Var(Fbar)` as a function of the seasonal cycle. Inputs `(season, year, member, lat, lon)` / `(season, year, lat, lon)` on the same grid → coefficient `(season, lat, lon)`. `temporal_sigma`: `None` (per-season, unsmoothed), a float (cyclic Gaussian smoothing across the season axis), or `"constant"` (a single pooled time-invariant slope — a pooled regression over all seasons, not the large-sigma limit). The probabilistic companion functions live in `deepscale.methods.smoothed_regression` — see `references/methods.md`.

## IO helpers

```python
def write_terciles(probs, path, *, title, method="") -> None
def tercile_mae(probs, reference) -> float
```

`write_terciles`: `(tercile,lat,lon)` fractional probs → NetCDF with `below`/`normal`/`above` **percent** (0-100) variables (renormalized, float32, `_FillValue=-9999.0`). `tercile_mae`: MAE in percentage points vs a reference (path to such a NetCDF or a percent DataArray); regrids the reference (linear) if grids differ; only cells finite on both sides in all three categories count.

## Public but not in `__all__`

- `deepscale.time.{season_step, season_bounds, season_months, season_times, infer_cadence, dekad_start_for, dekad_window, dekad_of_year, dekads_for_issuance, pentad_start_for, pentad_window}` — season-step alignment + dekad/pentad calendar arithmetic. Reference module-qualified (`deepscale.time.season_step`). See `references/analog-completion.md`.
- `deepscale.indices.REGIONS` (box definitions dict) and `deepscale.indices.Index.list_named()` — public, module-qualified only.
- `deepscale.metrics.leverage(x)` — simple-regression leverage `h_i`, alongside the top-level `loo_predict`/`loo_corr`; see `references/metrics-and-terciles.md`.
- `deepscale.tercile.{to_tercile, to_tercile_cv, cpt_tercile_forecast}` — see `references/metrics-and-terciles.md`.
- `deepscale.cv.{loyo, lko, blocked, expanding, get_cv}` — see `references/methods.md`.
- `deepscale.registry.{register_method, register_calibrator, register_metric, register_strategy, get_method, get_metric, get_strategy, ...}` — extension points.
- `deepscale.methods.base.{MethodBase, ProbabilisticMethodBase}` — subclass + `register_method` to add a method; `ProbabilisticMethodBase` adds `predict_distribution()`.
- `deepscale.logistic.logistic_forecast(...)` — low-level per-cell logistic engine.
- `deepscale.metrics.spread_error.spread_error_diagnostics(forecast, obs, *, spatial=False)`.
- `deepscale.methods.smoothed_regression.{fit_gamma, gamma_to_normal, normal_to_gamma, fit_ab, fit_ab_field, smooth_ab, normal_category_probs}` — the Kharin-2017 probabilistic calibration pipeline; see `references/methods.md`.
- `deepscale.metrics.crpss.{crps_normal, crps_climatology, crpss}` — Gaussian CRPS building blocks (ndarray in/out) behind the `crpss` metric.
- `deepscale.plotting.*` — see `references/plotting-reporting.md`.
