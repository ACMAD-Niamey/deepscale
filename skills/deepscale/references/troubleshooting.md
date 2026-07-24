# Troubleshooting: errors and environment setup

## Environment

- `pip install accord-deepscale`; `import deepscale as ds`. Python ≥ 3.10.
- Dev setup: `git clone`, `uv sync`. For real-data demos also clone rosetta alongside and `uv pip install -e ../rosetta` — the `[tool.uv] override-dependencies = ["zarr>=3.1.0"]` in pyproject exists precisely so rosetta (sheerwater pins `zarr==2.18.3`) and icechunk (`zarr>=3`) co-resolve.
- Extras: `plotting` (cartopy, matplotlib, rioxarray — needed for maps, PDFs, GeoTIFF), `dev` (pytest etc.), `validation` (reference downscaling libs). CorrDiff GPU deps (`torch`, `earth2studio`, `nvidia-physicsnemo`) are **not on PyPI extras** — install manually.
- Real-data examples need CDS credentials (`~/.cdsapirc`) with ERA5/C3S licences accepted — see the rosetta skill/README for setup.
- Tests: bare `pytest` runs the unit suite (< 30 s, coverage gate > 85% on `src/deepscale`); markers `integration` (real data), `agreement` (vs PyCPT reference), `gpu` are skipped unless requested: `pytest -m integration`.

## Error → cause table

| Symptom | Cause / fix |
|---|---|
| `ValueError` about grid mismatch from `calibrate`/`skill_compare` | These never regrid. `gcm.interp(lat=obs.lat, lon=obs.lon)` first |
| `ValueError` pointing to `to_tercile_cv()` from RPSS | You passed a continuous forecast to a probabilistic metric — convert to terciles (with CV discipline) first |
| `ValueError` from `pearson_r` on tercile input | Continuous metrics reject tercile forecasts — score the deterministic ensemble |
| `ValueError`: years not consecutive | CV schemes require integer years with gap 1. Usually caused by nesting `optimize()` inside an outer CV loop — use `train()` + `.predict()` instead |
| `ValueError` from `primary_metric="roc"` | `roc` expands to a dict; use a leaf metric (`roc_an`, `roc_area_below_normal`, ...) |
| `RuntimeError`: method requires training | `requires_training=True` (DL) method — `ds.train(name, ..., save_to=p)` then `ds.downscale(..., weights_path=p)` |
| `TypeError` passing both `gcm=` and `predictor_hindcast=` | `gcm=` is a deprecated alias; use `predictor_hindcast=` only |
| `RuntimeWarning`: ensemble gate fell back to uniform | Optimized weights didn't beat uniform under CV — working as designed; check `EnsembleResult.gate_passed` |
| `RuntimeWarning`: optimistic skill | You set `safeguards={"nested_cv": False}` — reported skill is no longer honest |
| `ValueError`: unknown safeguard key | `safeguards` accepts only `nested_cv`, `shrinkage`, `min_effective_n`, `gate` |
| `ImportError` naming `accord-deepscale[plotting]` | Install the plotting extra for maps/PDF/GeoTIFF |
| `NotImplementedError` from CorrDiff `save`/`load` | Torch model isn't picklable — re-instantiate instead of checkpointing |
| `NotImplementedError` from `flex_forecast(distribution="gamma")` | V1 is Gaussian-only |
| `NotImplementedError`: `transform_predictand="Gamma"` | CCA supports `None` or `"Empirical"` |
| `ValueError` from `prediction_error_variance` | `cv_predictions` and `obs` must cover the same set of years |
| `seasonal_mme` raises about years | Needs ≥ 5 intersection years across obs and all hindcasts |
| Whole regions NaN in tercile output | Degenerate boundaries (t33 == t67, e.g. dry cells) or uncalibratable cells (eReg < 3 finite years; logit < `min_years`) — masked by design |
| `ValueError`: crpss expects a Gaussian forecast with 'mu' and 'sigma' | The `crpss` metric takes an `xr.Dataset` of anomaly `mu`/`sigma`, not a field or tercile array — see `metrics-and-terciles.md` |
| `DeprecationWarning`: `Index.reduce(sst=...)` | The first positional argument is now `field` (the reduction is not SST-specific); pass `field=` — the old `sst=` still works for now. See `references/api.md` |
| `ValueError` from `complete`: climatology must have 'year' and 'step' dims | The analog source / climatology must be season-aligned — run it through `deepscale.seasonal_stack(...)` first. See `references/analog-completion.md` |
| `ValueError` from `percentile_of` / `frequency_below`: value must not / must carry the reduction dim | `percentile_of`/`rank_of_record` take a `values` array *without* `dim`; `frequency_below` takes a `sample` array *with* `dim`. See `references/analog-completion.md` |
| `KeyError`: Unknown index | `Index.named(...)` accepts `wvg, wvg2, nino12, nino3, nino34, nino4, oni, roni, dmi (iod), wtio, setio, wio, wpac` — call `deepscale.indices.Index.list_named()` for the current set |

If a "wrong result" (rather than an error) is the problem, first check the discipline rules in `metrics-and-terciles.md` — leakage and metric/forecast mismatches produce plausible-looking but invalid skill numbers.

## Known convention caveats

- **Empirical QM convention:** deepscale's empirical `qm` maps values directly through sorted historical columns (direct-CDF), whereas xsdba-style implementations apply *additive adjustment factors* (`ref_q − hist_q` interpolated). The conventions genuinely differ and direct-CDF is sample-hungry — with short training windows the parametric variant (`variant="parametric"`) or a longer window closes most of the gap. The default is kept for backward stability.
- **Series `quantile_map` clamps by default:** the top-level `deepscale.quantile_map` (scalar/series bias-correction, `references/analog-completion.md`) defaults to `extrapolate="clamp"`, which silently truncates a forecast of a record-breaking value down to the strongest event in the training record. Pass `extrapolate="linear"` (or `variant="parametric"`) when a new extreme must be allowed to exceed the historical range.

## Operational scripts (where to look; not covered in depth by this skill)

- `scripts/nightly/run_country.py` — per-country nightly pipeline (`python -m scripts.nightly.run_country`).
- `scripts/s2s/run_issuance.py` — sub-seasonal testbed (`python -m scripts.s2s.run_issuance`).
- `scripts/reproduce.py` — step-by-step PyCPT/CPT-Fortran parity reproduction (r ≈ 0.9996).
- `notebooks/experimentB1.ipynb` — PyCPT vs DeepScale side-by-side comparison.
