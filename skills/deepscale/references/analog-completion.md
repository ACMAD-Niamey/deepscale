# Analog selection, scenario completion, and climate positioning (SMPG)

The season-monitoring / seasonal-forecast-positioning (SMPG) subsystem: pick the
past years that resemble the one being forecast, splice a partly-observed season
forward with each of those years, and say where the result falls in the
historical record. Five modules, all pure xarray and all agnostic to what the
dimensions *mean* — the same call runs on a grid `(…, lat, lon)`, on an
admin-unit `(…, region)` aggregation, or on a single station series `(…,)`.

| Module | Top-level exports | Role |
|---|---|---|
| `deepscale.analog` | `AnalogSet`, `analogs_from_years`, `analogs_from_index`, `analogs_from_field`, `analogs_where` | *Which* past years are analogs |
| `deepscale.climate` | `seasonal_stack`, `seasonal_reduce`, `accumulate`, `percentile_of`, `frequency_below`, `rank_of_record` | Season aggregation + positioning a value in a record |
| `deepscale.completion` | `complete`, `CompletionResult` | Splice observed + forecast + analog remainders into scenarios |
| `deepscale.series` | `quantile_map`, `error_bounds`, `ErrorBounds` | Bias-correct / bracket a scalar forecast *series* |
| `deepscale.time` | (module-qualified only — see below) | Season-step alignment + dekad/pentad calendar arithmetic |

Typical flow: `seasonal_stack` an archive into `(year, step, …)` → select analog
years with `analogs_from_index`/`analogs_from_field`/`analogs_where` → `complete`
the current season → read `CompletionResult.percentile` / `rank_of_record` for
the headline. Runnable end-to-end: [../examples/analog_completion.py](../examples/analog_completion.py).

---

## Analog selection (`deepscale.analog`)

Every selector scores *all* candidate years (not just the chosen ones), so the
margin between the ninth and tenth analog is always inspectable. Every metric is
distance-like — **lower is a better analog** — including the correlation metrics,
which score `1 − r`; that uniformity is what lets `.top(n)`, `&` and `|` mean the
same thing regardless of how comparability was defined.

### `AnalogSet` (frozen dataclass)

```python
@dataclass(frozen=True)
class AnalogSet:
    years: np.ndarray          # selected years, best analog first
    scores: xr.DataArray       # distance-like score for EVERY candidate year, indexed by `year`
    metadata: dict             # selector name, metric, target value, ...
```

- `.candidates` (property) → every year that was scored, selected or not.
- `.top(n)` → the `n` best-scoring selected years (order preserved); raises if `n > len(years)`.
- `.weights(kind="uniform", *, scale=None)` → per-analog weights summing to 1, indexed by `year`.
  `kind` ∈ `"uniform"` (equal — the honest default when scores aren't calibrated distances),
  `"inverse_distance"` (`scale / (score + scale)`), `"gaussian"` (`exp(-½(score/scale)²)`).
  `scale` defaults to the median selected score; must be positive for the weighted kinds.
- `.filter(mask)` → drop selected years where the boolean `mask` (over `year`) is False.
- `len()`, iteration (over years), and `repr` are defined.
- Set algebra: `a & b` (years selected by both) and `a | b` (years selected by either),
  both **re-ranked on the mean of the two score vectors** aligned on shared candidate years
  (a year scored by only one side keeps that side's score, so a union stays rankable);
  raises if the two sets share no candidate years.

### Constructors

```python
analogs_from_years(years, *, candidates=None, scores=None) -> AnalogSet
```
Explicit list (expert judgement). Order given is order kept, no metric. `candidates` (or
`scores`) supplies the pool so the result still reports what was *not* chosen.

```python
analogs_from_index(index, target=None, *, target_year=None, n=None,
                   metric="absolute", candidates=None) -> AnalogSet
```
Rank years by how close their scalar `index` value (a series over `year`, e.g.
`Index.reduce(...)`) is to `target` (a value) or to the value `index` takes in `target_year`
(**exactly one** of the two required). `metric` ∈ `"absolute"`, `"squared"`, `"signed"`
(`"signed"` ranks by `target − index`, so only years *at or beyond* the target score well).
`n=None` keeps every scored year, ranked. `target_year` is **not** excluded automatically
(it is its own perfect analog) — pass `candidates` to leave it out.

```python
analogs_from_field(field, target=None, *, target_year=None, n=None,
                   metric="rmse", region=None, weights="cos_lat", candidates=None) -> AnalogSet
```
Rank years by how closely their `(year, lat, lon)` `field` *pattern* resembles a target
`(lat, lon)` pattern (or the pattern in `target_year`; **exactly one** required). A `member`
dim is averaged out. `metric` ∈ `"rmse"`, `"mae"`, `"correlation"` (centres each pattern on its
own spatial mean), `"anomaly_correlation"` (anomalies about the climatology, no further
spatial centring — this is what distinguishes ACC from spatial Pearson). `region` = bbox
`[lat_s, lat_n, lon_w, lon_e]` (or a shapefile/geometry, which requires Rosetta). `weights` ∈
`"cos_lat"` (default — otherwise tall regions are dominated by high-latitude cells), `None`, or
an `xr.DataArray`. Nothing here is SST-specific.

```python
analogs_where(condition, *, scores=None) -> AnalogSet
```
Select every year where the boolean `condition` (over `year`) holds — for threshold criteria
rather than distances, e.g. `analogs_where(nino34 >= 0.5)`. Raises if the condition selects no
years. Pass `scores` to supply a ranking; by default selected years are equally good, and
composing with `&`/`|` against a distance-based set restores an ordering.

```python
strong = analogs_where(nino34 >= 0.5)
rapid  = analogs_where((nino34 - nino34_spring) >= 1.0)
analogs = (strong & rapid).top(9)      # strong El Niño AND rapid onset
```

---

## Climate positioning (`deepscale.climate`)

### Season aggregation

```python
seasonal_stack(da, season, *, time_dim="time", cadence=None, years=None) -> (year, step, …)
```
Reshape a continuous `(time, …)` series into one season per year, `step` being the ordinal
position within the season — the layout that makes years comparable (step 3 of 1997 and step 3
of 2026 are the same point in the season, which their calendar timestamps never are). `season`
is anything `deepscale.time.season_bounds` accepts (`"JJAS"`, `(10, 2)`, timestamp pair).
`step` is long enough for the longest season in `years` (they differ by a day across leap years);
adds a `season_start` coord on `year`. Years with no data in the season drop; partial years are
kept and NaN-padded. **This is the input `complete` expects for `climatology`.**

```python
seasonal_reduce(da, months, *, how="sum", time_dim="time") -> (year, …)
```
Select calendar `months` (list of 1–12) and reduce each year's timesteps to one value along
`time_dim`, replacing it with `year`. `how` ∈ `"sum"`, `"mean"`, `"max"`, `"min"`. E.g.
`seasonal_reduce(precip, [10, 11, 12])` = each year's OND total.

```python
accumulate(da, *, window=None, dim="time", how="sum", min_count=None) -> …
```
Accumulate along `dim`. `window=None` collapses `dim` entirely (one total); an int is a trailing
rolling window (keeps `dim`, each stamp holds the accumulation ending there). `how` ∈
`"sum"/"mean"/"max"/"min"`. `min_count` = minimum non-NaN steps to produce a value; defaults to
requiring **every** step (so a partially-missing accumulation is NaN, not a silent under-count) —
pass `1` to accumulate whatever is present.

### Positioning a value in a record

All three reduce over one dim (`dim="year"` by default) and preserve every other dim, and all
restore NaN where the input value is NaN (a comparison like `clim < value` would otherwise report
percentile/frequency 0 for missing data).

```python
percentile_of(values, climatology, *, dim="year", method="empirical") -> [0,1]
```
Where `values` (must **not** carry `dim`) falls in `climatology`'s distribution along `dim`.
`method` ∈ `"empirical"` (mid-rank: fraction strictly below + half the tied fraction; bounded,
assumption-free), `"weibull"` (`rank/(n+1)` plotting position — never exactly 0 or 1),
`"gaussian"` (fits a normal and evaluates its CDF — extrapolates but assumes symmetry).

```python
frequency_below(sample, climatology, *, q=1/3, dim="year") -> [0,1]
```
Fraction of `sample` (carries `dim`) below the `q`-th quantile of `climatology` (carries `dim`).
With `q=1/3` and each analog year's seasonal total as `sample`, this is the **below-normal
frequency composite**: the share of analog years that landed in the dry tercile at each location.
Below-only by design — for a frequency-*above* composite, flip the sign of both arrays (or use
`1 − frequency_below(...)` when there are no ties on the threshold). NaN `sample` entries are
excluded from the fraction; a cell with no valid `climatology` year returns NaN.

```python
rank_of_record(values, climatology, *, dim="year", ascending=True) -> rank
```
Rank of `values` within `climatology ∪ {values}` along `dim`. `ascending=True` (default) → rank 1
is smallest, so `rank_of_record(...) == 1` reads "driest on record" for rainfall; `ascending=False`
→ rank 1 is largest. Ties share the better (lower) rank (competition ranking). A brand-new value's
maximum possible rank is `n + 1`.

---

## Scenario completion (`deepscale.completion`)

```python
complete(observed, analogs, *, climatology, season, year=None, forecast=None,
         analog_source=None, reduce="median", weights=None, percentile_reference=None,
         percentile_method="empirical", overlap="observed", time_dim="time",
         cadence=None, min_count=None) -> CompletionResult
```

Splices, on the season's **ordinal step** (never on calendar dates): observations to date, then
the forecast, then each analog year's remainder — one plausible end-of-season outcome per analog.
The spread across analogs *is* the uncertainty, and every member is a real season that happened.
`complete` reduces along `step` and touches nothing else, so `(time, lat, lon)` → per-pixel maps,
`(time, region)` → per-district curves, `(time,)` → a single series. There is no separate
admin-unit code path.

- `observed` — `(time, …)` increments so far, or `None` for a pure analog projection (the useful
  null case: the season on history alone). At least one of `observed`/`forecast` is required.
- `analogs` — an `AnalogSet` (however selected).
- `climatology` — `(year, step, …)` season-aligned increments from `seasonal_stack`; supplies the
  analog remainders and (by default) the `percentile` reference. `analog_source` overrides the
  remainder source (e.g. a bias-corrected archive) if it differs from `climatology`.
- `season`, `year` — the accumulation window (`season` per `deepscale.time.season_bounds`);
  `year` defaults to the season year of the first observed (or forecast) stamp.
- `forecast` — `(time, …)` increments for the steps after the observations. **Omit it and run
  twice to isolate exactly what the dynamic forecast contributes** — the comparison a forecaster
  needs before trusting it.
- `reduce` — how scenario totals become one `consensus`: `"median"` (default; the tails are one
  year each), `"mean"`, a float quantile, or a callable `f(totals, dim="scenario")`.
- `weights` — per-analog: `None` (equal), a string passed to `AnalogSet.weights`
  (`"uniform"/"inverse_distance"/"gaussian"`), or an explicit `xr.DataArray`.
- `overlap` ∈ `"observed"` (default), `"forecast"`, `"error"` — which segment wins where
  observations and forecast cover the same step.

```python
@dataclass(frozen=True)
class CompletionResult:
    scenarios: xr.DataArray    # (scenario, step, …) per-step increments; one scenario per analog year
    totals: xr.DataArray       # (scenario, …) season totals — one plausible outcome per analog
    consensus: xr.DataArray    # (…) totals reduced across scenarios (median by default)
    percentile: xr.DataArray   # (…) where consensus falls in the record, [0,1]
    segments: xr.DataArray     # (step,) of "observed" / "forecast" / "analog"
    analogs: AnalogSet         # the selection used, so a result can explain itself
    metadata: dict
    def accumulation(self) -> xr.DataArray          # (scenario, step, …) running season-to-date totals
    def segment_steps(self, segment) -> np.ndarray  # step indices contributed by one segment
```

---

## Scalar-series calibration (`deepscale.series`)

Everything else in deepscale calibrates a *field*; these two calibrate a single number per year
(a Niño3.4 value, a region-averaged rainfall total, a basin-mean SST). Neither knows what the
series measures.

```python
quantile_map(x, source, target, *, variant="empirical", extrapolate="clamp")
```
Map `x` from the distribution of `source` onto that of `target` — canonically bias-correcting a
forecast index (`source` = model hindcast series, `target` = observed series, `x` = the new
forecast). Uses the same transfer function as the gridded `qm` downscaler
(`deepscale.methods._qm_kernel`), so the two cannot disagree. `x` may be a scalar, array, or
`xr.DataArray` (dims/coords preserved); `source`/`target` need not be the same length.
`variant` ∈ `"empirical"` (matches sample CDFs) or `"parametric"` (matches Gaussian fits — more
stable on short records, assumes symmetry). `extrapolate` ∈ `"clamp"` (default — **silently
truncates a record-breaking forecast to the strongest training event**; see troubleshooting) or
`"linear"` (continues the transfer function's end slope); ignored by the parametric variant.

```python
error_bounds(hindcast_prediction, hindcast_obs, forecast, *,
             level=0.8, dim="year", method="empirical") -> ErrorBounds
```
A `level`-confidence interval around `forecast` built from the hindcast's realised errors
(`hindcast_prediction − hindcast_obs`). Because an observed value is `prediction − error`, the
interval is the forecast minus the error distribution's tail quantiles — which also removes any
mean bias for free. `hindcast_prediction`/`hindcast_obs` are paired over `dim` (other dims
broadcast, so a gridded or multi-model interval falls out of the same call); `forecast` must not
carry `dim`. `method` ∈ `"empirical"` (tail quantiles off the sample; needs ≥ 2 paired years) or
`"gaussian"` (fits a normal — extrapolates into the tails, assumes symmetry).

```python
@dataclass(frozen=True)
class ErrorBounds:
    lower; upper; level: float; bias; errors: xr.DataArray
    # unpacks as (lower, upper): `lo, hi = error_bounds(...)`
```

---

## Calendar utilities (`deepscale.time`)

Reference these **module-qualified** — `time` is imported as a submodule but its functions are
**not** in the top-level `__all__` (use `deepscale.time.season_step`, not `ds.season_step`).

**Season-step alignment** — the coordinate every cross-year splice in deepscale joins on:

```python
season_step(time, season, *, year=None, cadence=None) -> xr.DataArray   # 0-based ordinal in season; -1 outside
season_bounds(season, year) -> (start, end)     # inclusive Timestamps. season = "JJAS" | (10, 2) | (ts, ts)
season_months(season) -> list[int]              # "JJAS" -> [6,7,8,9]; wraps ("NDJ" -> [11,12,1])
season_times(season, year, cadence) -> pd.DatetimeIndex   # start stamp of every step, in order
```

`season` codes are month-initial strings resolved by contiguity (`"JJAS"`, `"OND"`), an
`(start_month, end_month)` int pair (wraparound like `(10, 2)` ends in `year + 1`), or a
timestamp pair.

**Cadence inference:**

```python
infer_cadence(time) -> "daily" | "pentad" | "dekad" | "monthly"   # ValueError on 1 stamp or seasonal/annual spacing
```

**Dekad / pentad arithmetic** (a dekad is a 10-day calendar slice — day 1–10, 11–20, 21–end — so
the third dekad is 8–11 days long; a pentad is a fixed 5-day slice; each is identified by its
start date):

```python
dekad_start_for(d) -> date          dekad_window(start) -> (date, date)
dekad_of_year(d) -> int             dekads_for_issuance(issuance, lead_days=(lo, hi)) -> list[date]
pentad_start_for(d) -> date         pentad_window(start) -> (date, date)
```
