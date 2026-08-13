# DeepScale

Modular downscaling, calibration, and verification for seasonal climate forecasts.

DeepScale turns coarse global-model (GCM) forecasts and fine-resolution observations into calibrated, high-resolution forecast products, and scores them with cross-validated skill metrics. It operates on xarray arrays and is agnostic to where the data came from, so it pairs naturally with a data layer like [Rosetta](https://github.com/accord-research/rosetta) but does not require it.

Downscaling methods, skill metrics, and ensemble strategies are looked up by name from a registry, so you select them with plain strings and can add new ones without changing the orchestration code.

## Installation

```bash
pip install accord-deepscale
```

The distribution is published as `accord-deepscale`; the import name is `deepscale`:

```python
import deepscale
```

The shapefile and region-clipping helpers additionally require Rosetta:

```bash
pip install accord-rosetta
```

DeepScale requires Python 3.10 or newer.

## Core API

```python
import deepscale

# Bias-correct and downscale one model against observations.
result = deepscale.downscale(gcm, obs, method="bcsd")

# Turn a predictor into below/normal/above tercile probabilities.
probs = deepscale.calibrate(predictor, obs, method="ereg")

# Try several methods and keep the most skillful.
best = deepscale.optimize(gcm, obs, methods=["bcsd", "cca"])

# Combine multiple models into one forecast.
mme = deepscale.ensemble([model_a, model_b], obs, strategy="uniform")

# Score a forecast against observations.
report = deepscale.skill(forecast, obs, metrics=["rpss", "roc"])
```

Inputs are xarray arrays with CF-style coordinates. A GCM hindcast has dimensions `(year, member, lat, lon)` and observations have `(year, lat, lon)`. Outputs are continuous or tercile forecast products plus skill summaries and maps. Terciles are ordered `[0, 1, 2]` for below-normal, normal, and above-normal.

## What is included

Everything below is selected by name.

Downscaling and bias-correction methods, passed as `method=` to `downscale()` and `optimize()`:

| Method | Description |
|---|---|
| `bcsd` | Bias correction with spatial disaggregation |
| `cca` | Canonical correlation analysis |
| `qm` | Quantile mapping |
| `dqm` | Detrended quantile mapping |
| `delta` | Delta-change |
| `climatology` | Climatological baseline |
| `rank-analog` | Rank-based quantile matching |
| `corrdiff` | NVIDIA CorrDiff diffusion downscaling; needs GPU dependencies that are not on PyPI (see `src/deepscale/methods/corrdiff.py`) |

Calibration methods, passed as `method=` to `calibrate()`:

| Method | Description |
|---|---|
| `ereg` | Ensemble regression |
| `logit` | Logistic index calibration |
| `smoothed_regression` | Kharin et al. (2017) smoothed-coefficient calibration; season-aware, with deterministic and tercile-probability output |

Ensemble strategies, passed as `strategy=` to `ensemble()`: `uniform`, `skill_weighted`, `bma`, `drop_worst`.

Skill metrics, passed as `metrics=` to `skill()`: `rpss`, `roc`, `roc_area_below_normal`, `roc_area_above_normal`, `generalized_roc`, `pearson_r`, `spearman`, `2afc`, `root_mean_squared_error`, `mean_square_skill_score` (`msss`), `continuous_ranked_probability_skill_score` (`crpss`), `heidke_skill_score`, `reliability`, `spread_error_ratio`, `spread_error_correlation`.

Cross-validation schemes: `loyo` (leave-one-year-out), `lko` (leave-k-out), `blocked`, `expanding`.

## Daily aggregations

Rainy-season timing and dry-spell statistics, for products that need a date rather than a seasonal total. These take continuous daily rainfall rather than the seasonal arrays above, and are module-qualified rather than selected by name:

```python
from deepscale import aggregations as agg

onset = agg.onset(daily, season="MAM")                       # default criterion
cessation = agg.cessation(daily, season="MAM", after=onset)
length = agg.season_length(onset, cessation)
spells = agg.dry_spell(daily, season="MAM")
```

| Function | Returns |
|---|---|
| `onset` | first day of the rainy season, as days since the season start, with the resolved calendar date |
| `cessation` | first qualifying dry spell after a given point, usually onset |
| `season_length` | days from onset to cessation |
| `dry_spell` | longest dry run in the season, and how many runs reached the qualifying length |

Onset defaults to 20 mm across 3 consecutive days, rejected as a false start if a 7-day dry spell falls within the following 21 days. Every threshold is a keyword argument, so other regional definitions are one call away, and the values used are recorded on the output for provenance.

Timing results carry a three-state `occurred` field distinguishing a season that failed from a cell with no data, which a NaN date alone cannot express. Output dims are `(year, lat, lon)`, the same shape the rest of the library takes for observations. Full detail, including how much daily data each function needs past the season end, is in [skills/deepscale/references/aggregations.md](skills/deepscale/references/aggregations.md).

## Example workflow

The repository ships a runnable end-to-end demo:

```bash
python examples/demo_forecast.py
```

It uses Rosetta to fetch ERA5 temperature observations (`obs/era5`) and ECMWF seasonal hindcasts (`c3s/ecmwf-monthly`), reshapes them into DeepScale inputs, then runs optimize, tercile conversion, and skill scoring. Rosetta handles the remote retrieval and normalization; DeepScale starts from the prepared xarray datasets.

The demo needs CDS credentials in `~/.cdsapirc` with the relevant dataset licences accepted (see the Rosetta README for setup). `examples/README.md` lists all demos and their prerequisites.

## Calibration

`deepscale.calibrate()` produces tercile probabilities with dims `(tercile, lat, lon)` directly. Use it when the predictor is already on the target grid, or when a scalar index drives the forecast.

### Ensemble regression (`method="ereg"`)

eReg fits each model independently with per-grid-cell ordinary least squares: the ensemble-mean hindcast predicts the observed field, and the chosen forecast year is converted to parametric tercile probabilities. Multiple models are averaged after each produces its own probability map.

```python
probs = deepscale.calibrate(
    {
        "ecmwf": (ecmwf_hindcast_on_obs_grid, ecmwf_forecast_on_obs_grid),
        "ukmo": (ukmo_hindcast_on_obs_grid, ukmo_forecast_on_obs_grid),
    },
    obs,
    method="ereg",
    forecast_year=2026,
)
```

Each hindcast needs a `year` dimension, an optional `member` dimension, and spatial dimensions named `lat/lon`, `latitude/longitude`, `Y/X`, or `y/x`. eReg calibrates, it does not regrid, so put model fields on the observation grid first. If every forecast contains exactly one year, `forecast_year` is inferred.

### Logistic index calibration (`method="logit"`)

logit fits a gridded logistic relationship between a scalar predictor index and observed tercile occurrence. Pass the hindcast index series as `predictor` and the forecast-year value as `forecast`.

```python
index = deepscale.Index.named("wvg")
hindcast_index = index.reduce(sst_hindcast)
forecast_index = index.reduce(sst_forecast, climatology=sst_hindcast)

probs = deepscale.calibrate(
    hindcast_index, obs, method="logit", forecast=forecast_index,
)
```

For gridded SST predictors, `LogitConfig` reduces the fields through an `Index` before calibration:

```python
probs = deepscale.calibrate(
    predictor_hindcast=sst_hindcast,
    predictor_forecast=sst_forecast,
    obs=obs,
    method=deepscale.LogitConfig(
        index=deepscale.Index.named("wvg"),
        detrend=True,
        significance=0.1,
    ),
)
```

### Smoothed-coefficient regression (`method="smoothed_regression"`)

Implements the postprocessing method of Kharin, Merryfield, Boer & Lee (2017, *Mon. Wea. Rev.* 145, 3545–3561). It rescales the ensemble-mean anomaly with a per-grid-cell regression coefficient, but where `ereg` fits each season independently, `smoothed_regression` smooths the coefficients *across the seasonal cycle* to suppress the sampling error that a ~30-year record leaves in each season's estimate. This recovers, and often improves, skill in weakly predictable regimes where naive per-season calibration degrades it.

It is season-aware: inputs carry a `season` dimension (up to 12 rolling seasons) that this method owns. `temporal_sigma` sets the smoothing: `None` per-season, a float for cyclic Gaussian smoothing across the calendar, or `"constant"` for a single year-round coefficient. It is fit-and-apply (cross-validation is the caller's concern, as with `ereg`).

Two output modes via `output_type`:

```python
# Deterministic: rescaled ensemble-mean anomaly (scored with the `msss` metric).
adjusted = deepscale.calibrate(
    hindcast, obs,                       # (season, year, member, lat, lon) / (season, year, lat, lon)
    method="smoothed_regression",
    output_type="deterministic",
    temporal_sigma="constant",
    forecast_year=2024,
)

# Probabilistic: below/normal/above tercile probabilities (scored with `crpss`, `reliability`).
probs = deepscale.calibrate(
    hindcast, obs,
    method="smoothed_regression",
    output_type="tercile",
    distribution="gamma",                # "normal" for temperature, "gamma" for precipitation
    forecast_year=2024,
)
```

The probabilistic mode additionally calibrates the forecast spread and, for precipitation, works through a gamma distribution so probabilities never fall on negative rainfall. `deepscale.seasonal_coefficients(hindcast, obs, temporal_sigma=...)` exposes the fitted, smoothed coefficient field for inspection or plotting.

Runnable examples: `examples/demo_ensemble_regression.py` (eReg) and `examples/demo_logistic_wvg.py` (logit).

## Relationship to Rosetta

Rosetta handles data acquisition and normalization; DeepScale handles forecasting and verification. The interface between them is standardized xarray, so DeepScale stays source-agnostic and works with any data prepared the same way.

## Development setup

```bash
git clone https://github.com/accord-research/deepscale.git
cd deepscale
uv sync
```

Some examples also use Rosetta for data acquisition:

```bash
cd ..
git clone https://github.com/accord-research/rosetta.git
cd rosetta
uv sync
```

The roadmap (PyCPT parity, additional methods, and machine-learning tiers) is tracked on [GitHub Issues](https://github.com/accord-research/deepscale/issues?q=is%3Aopen+label%3Av1-roadmap) under the `v1-roadmap` label.
