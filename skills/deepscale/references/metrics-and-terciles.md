# Skill metrics and tercile conversion

## Tercile conversion — the leakage discipline

Two distinct paths in `deepscale.tercile`. Choosing the wrong one silently inflates skill.

### `to_tercile(forecast, obs_climatology, method="counting")` — production path

- `forecast` `(member, lat, lon)`; boundaries = 1/3 and 2/3 empirical quantiles of the supplied climatology over `year`.
- `method`: `"counting"` (fraction of members per category) or `"gaussian"`.
- Returns `(tercile, lat, lon)`.
- **Leakage hazard:** for CV hindcasts, passing the full obs leaks the held-out year into the boundaries. Inside a manual CV loop use `to_tercile(pred, obs.sel(year=train_years))` at most — or better, use `to_tercile_cv`.

### `to_tercile_cv(cv_predictions, obs, method="bootstrap", leverages=None, n_modes=3, cpt_boundaries=True)` — scoring path

- `cv_predictions` `(year, [member,] lat, lon)` from held-out folds; returns `(year, tercile, lat, lon)` with held-out discipline.
- `method`: `"cpt"` (Student-t with leverage-inflated PEV, `dofr = n − n_modes − 1`; requires per-year `leverages` from a CCA fit), `"bootstrap"`, `"gaussian_loo"`, `"t"`, `"gaussian_pev"`.

### `cpt_tercile_forecast(forecast, t33, t67, s2, dofr, leverage=0.0)`

Single-map CPT Student-t tercile kernel (the same math CPT Fortran uses for its probabilistic forecast page).

### Boundaries

Default is CPT-compatible empirical quantiles (`rndx = n*p + 0.5` convention). Cells where t33 == t67 (degenerate/dry) get NaN boundaries and drop out of skill computations — expected behavior.

## Metrics (`metrics=` to `skill()`)

All metrics are classes with `compute(forecast, obs, spatial=False, **kwargs)`; `spatial=False` → scalar (pooled over cells and years), `spatial=True` → `(lat, lon)` map. Some add `compute_diagram(...)`, captured in `SkillReport.diagrams`.

### Probabilistic (require `tercile` dim of size 3)

| Name (aliases) | Range / no-skill | Semantics |
|---|---|---|
| `rpss` | (−∞, 1]; 0 = climatology | Ranked Probability Skill Score vs climatology `[1/3, 2/3, 1]`; CPT categorization (strict `<`). kwargs: `loo_boundaries=False`, `bounded=False`, `cv_window=1`. Raises `ValueError` (pointing to `to_tercile_cv`) if no tercile dim |
| `roc` | [0,1] per category; 0.5 = no skill | Returns a dict `{"roc_bn", "roc_nn", "roc_an"}` (AUC per category). Diagram: `{bn/nn/an: {fpr, tpr, area}}`. **Not usable as `primary_metric`** — use the leaf metrics below |
| `roc_area_below_normal`, `roc_area_above_normal` | [0,1]; 0.5 | Scalar per-category AUCs (leaf metrics) |
| `generalized_roc` (`groc`) | [0,1]; 0.5 | Multi-category discrimination. `loo_boundaries` kwarg; NaN + warning if < 2 distinct obs categories |
| `reliability` | ≥ 0; 0 = perfect | Brier reliability decomposition term (lower is better), `n_bins=5`. Diagram: per-category bins |
| `heidke_skill_score` (`hss`) | ≤ 1; 0 = chance | Categorical: tercile collapsed via argmax, pooled contingency table (or per-cell when spatial) |

### Continuous / deterministic (reject a `tercile` dim; need the ensemble or its mean)

| Name (aliases) | Notes |
|---|---|
| `pearson_r` | Anomaly correlation over `year`; averages `member` first; raises `ValueError` on tercile input |
| `spearman` | Rank correlation, NaN-aware |
| `2afc` | P(forecast correctly ranks two random non-tied obs years); [0,1], 0.5 = no skill |
| `root_mean_squared_error` (`rmse`) | RMSE |
| `mean_square_skill_score` (`msss`) | 1 − MSE/Var(O); 1 = perfect, 0 = no better than the climatological mean, < 0 = worse than climatology. Averages `member` first; raises `ValueError` on tercile input. Pooled (`spatial=False`) averages the MSE and Var(O) fields separately before dividing (area-aggregated convention). Pairs with `smoothed_regression` `output_type="deterministic"` |
| `spread_error_ratio` | `mean(spread)/mean(error)`; ≈1 = well calibrated. Needs `member` dim |
| `spread_error_correlation` | Pearson r between per-year spread and error; > 0 desirable; NaN + warning if < 3 years. Needs `member` dim |

Helper: `deepscale.metrics.spread_error.spread_error_diagnostics(forecast, obs, *, spatial=False) -> SpreadErrorDiagnostics(spread, error)`.

### Parametric-Gaussian probabilistic (neither table above — a distinct input contract)

| Name (aliases) | Range / no-skill | Semantics |
|---|---|---|
| `continuous_ranked_probability_skill_score` (`crpss`) | (−∞, 1]; 0 = climatology | CRPS skill score for a parametric Gaussian forecast against the climatological `N(0, std(obs))` reference. `forecast` is an `xr.Dataset` with `mu` and `sigma` data-vars over `(year[, lat, lon])` anomalies (NOT a `tercile` field); raises `ValueError` otherwise. Closed-form Gaussian CRPS (Gneiting et al. 2005). Building blocks `crps_normal` / `crps_climatology` / `crpss` are exposed on `deepscale.metrics.crpss` |

### Presets

- `metrics="svslrf"` → `["rpss", "roc", "reliability"]` (the WMO-SVSLRF mandatory triplet).
- `metrics="all"` → every registered metric; ones that raise `ValueError` on incompatible shapes are skipped with a `RuntimeWarning`.

### Mask discipline and self-checks (comparing several forecasts)

RPSS masks its climatology reference where **obs** is NaN — not where the *forecast* is NaN. When forecasts with different NaN footprints (different models, methods, or regrid artifacts) are scored against the same obs, each one is silently compared over a different cell set, and the numbers drift: in one real case a uniform-1/3 (zero-information) forecast scored RPSS **+0.26** purely from mask mismatch. Two rules, learned the hard way:

1. **One common valid mask before scoring.** Build it once — cells where the tercile probs are finite and sum to ~1 in *every* forecast being compared, intersected with cells where the obs tercile boundaries are defined — and `.where(mask)` every forecast *and* the obs before any `skill()` call.
2. **Self-check every scoring run.** A uniform `[1/3, 1/3, 1/3]` forecast must score RPSS ≈ 0 (assert `|RPSS| < 0.02`), and a perfect forecast ≈ 1. If the climatology check fails, the masks are mismatched — fix that before trusting any other number.

### Pairing rule

Score tercile metrics on the tercile forecast and continuous metrics on the deterministic ensemble — two separate `skill()` calls:

```python
report_prob = ds.skill(cv_terciles, obs,
                       metrics=["rpss", "hss", "roc", "generalized_roc", "reliability"],
                       spatial=True)
report_det = ds.skill(cv_continuous, obs,
                      metrics=["pearson_r", "spearman", "2afc", "rmse",
                               "spread_error_ratio", "spread_error_correlation"],
                      spatial=True)
```

## Significance and multiple-comparison control

Top-level exports for deciding whether a discovered predictor–predictand relationship is more than the best of a large search — e.g. screening many candidate teleconnection indices against a rainfall series. These are plain functions on paired 1-D series (not registered `MetricBase` scores).

**Leave-one-out CV helpers** (`deepscale.metrics.cross_validation`, re-exported at top level except `leverage`):

```python
ds.loo_predict(x, y)                    # closed-form LOO predictions of y ~ x (Allen's PRESS identity)
ds.loo_corr(x, y, *, min_finite=5)      # corr(LOO predictions, y): 1 perfect, ~0 none, NEGATIVE for chance
deepscale.metrics.leverage(x)           # per-point leverage h_i = 1/n + (x-x̄)²/Sxx (module-qualified only)
```

`loo_corr` is *negative* for a no-skill predictor (LOO predictions of a near-zero relationship anti-correlate with the target). This is exactly why the permutation test below defaults to a **one-sided upper tail**. Each returns NaN on a degenerate (zero-variance) fit; `loo_corr` also NaNs with fewer than `min_finite` finite pairs or a constant prediction.

**Significance** (`deepscale.metrics.significance`, re-exported at top level):

```python
ds.permutation_test(x, y, *, statistic=loo_corr, alternative="greater", n=5000, rng=None)
    -> (observed_statistic, p_value)
ds.fdr(pvalues, *, method="bh") -> array of q-values (same shape as input)
```

`permutation_test` holds `x` fixed, shuffles `y` `n` times, and returns the fraction of shuffles at least as extreme as the observed statistic, with the conventional `+1` in numerator and denominator so it is never exactly zero (`p_value` is NaN if the observed statistic is not finite). `statistic` is any `f(x, y) -> float`; the default `loo_corr` is why `alternative` defaults to `"greater"` (`"less"` / `"two-sided"` also accepted — but a two-sided `|stat|` test is *invalid* for a LOO skill statistic whose null is biased negative). `rng` is a seed or `numpy.random.Generator` (share one generator across calls for a stable draw sequence).

`fdr` returns Benjamini–Hochberg q-values (`q_i` = smallest FDR at which test `i` is rejected). NaN p-values are excluded from the test count and returned as NaN, so a ragged leaderboard passes through directly. Only `method="bh"` is implemented.
