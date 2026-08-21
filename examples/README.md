# DeepScale examples

Runnable demo scripts for the DeepScale forecasting API. Each script is
self-contained and is run from the repository root with `uv run`:

```bash
uv sync                                       # installs deepscale into the env
uv run python examples/demo_quantile_mapping.py
```

The scripts in the first table below run offline. The real-data scripts also
need Rosetta (a separate repo) installed into the same environment:

```bash
uv pip install -e ../rosetta                  # clone rosetta alongside deepscale first
```

## Run offline (no network, no credentials)

| Command | Shows |
| --- | --- |
| `uv run python examples/demo_quantile_mapping.py` | Quantile-mapping bias correction, empirical and parametric |
| `uv run python examples/demo_detrended_qm.py` | Detrended QM vs plain QM: preserving a warming trend |
| `uv run python examples/demo_delta_scaling.py` | Delta-scaling baseline: GCM anomaly onto the obs climatology |
| `uv run python examples/demo_checkpoint_roundtrip.py` | Save a fitted method, reload it, reproduce the prediction bit-for-bit |
| `uv run python examples/demo_train_inference.py` | Separated train / inference, with a custom registered method |
| `uv run python examples/demo_probabilistic_method.py` | A custom probabilistic method scored by counting tercile members |
| `uv run python examples/demo_ensemble_regression.py --synthetic` | `calibrate(method="ereg")`: ensemble-regression tercile probabilities |
| `uv run python examples/demo_smoothed_forecast.py` | `calibrate(method="smoothed_regression")`: real-time out-of-sample forecast, pooled multi-model super-ensemble |
| `uv run python examples/demo_logistic_wvg.py --synthetic` | `calibrate(LogitConfig(...))`: WVG-index logistic tercile probabilities |
| `uv run python examples/seasonal_forecast_eastafrica_mam.py --dry-run` | The full multi-phase MME pipeline, plan only (`--tiny` runs it on synthetic data) |

## Need real data or a GPU

These fetch observations and hindcasts through Rosetta, so they need CDS
credentials in `~/.cdsapirc` and the relevant dataset licences accepted. The
CorrDiff demo needs an NVIDIA GPU.

| Command | Shows | Needs |
| --- | --- | --- |
| `uv run python examples/demo_ensemble_regression.py` | eReg on real C3S + ERA5 (drop `--synthetic`) | CDS |
| `uv run python examples/demo_logistic_wvg.py` | WVG logistic on real ERA5 (drop `--synthetic`) | CDS |
| `uv run python examples/demo_forecast.py` | End-to-end: optimize, leave-one-year-out skill, plot | CDS |
| `uv run python examples/demo_seasonal_mme.py` | Seasonal multi-model ensemble pipeline | CDS |
| `uv run python examples/demo_seasonal_mme_multimodel.py` | Multi-model MME with per-member contributions | CDS |
| `uv run python examples/demo_realdata_comparison.py` | Compare downscaling methods on real CHIRPS + C3S | CDS |
| `uv run python examples/demo_realdata_skill.py` | Leave-one-year-out RPSS skill of each method | CDS |
| `uv run python examples/demo_chelsa.py` | CHELSA V2 over Rwanda: real C3S/CHIRPS, GLO-90 terrain, ERA5 wind/PBL, official exposure, conservation checks, PNG + NetCDF | CDS + CHELSA/GLO-90 HTTP |
| `uv run python examples/seasonal_forecast_eastafrica_mam.py --phase 0 1 2` | The full PyCPT-parity reference forecast | CDS |
| `uv run python examples/demo_corrdiff.py` | NVIDIA CorrDiff diffusion downscaling | GPU |

Outputs (PNG, NetCDF) are written to `examples/output/`, which is git-ignored.
Tercile maps draw coastlines and borders when `cartopy` or `geopandas` is
available, and fall back to a plain map otherwise.
