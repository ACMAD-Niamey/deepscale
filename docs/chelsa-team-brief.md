# CHELSA precipitation downscaling in DeepScale

> **Bottom line.** CHELSA is a physically motivated way to redistribute coarse-grid precipitation across mountains and valleys. We reproduced the published equations in DeepScale and verified the important numerical properties. In our East Africa tests it added useful detail and performed especially well in Rwanda's highest terrain, but it did **not** improve national-average accuracy. It should remain an optional method, not a default or a substitute for local validation.

## The idea in one minute

A seasonal climate model may say how much rain falls over a large grid cell, but not how that rain is distributed among the ridges, slopes, and valleys inside it. CHELSA adds that local structure.

It asks three intuitive questions:

1. Is this location on the **windward** side of terrain, where rising air can favor rain, or in a **lee-side rain shadow**?
2. Is the location near or above the **planetary boundary layer**—the lower part of the atmosphere most directly affected by the surface?
3. Is it an **exposed ridge** or a sheltered valley?

Those factors become a relative wet/dry weight for every fine-grid cell. The weights are then normalized inside each coarse model cell, so CHELSA changes **where** the rain falls without changing the coarse model's total.

<div class="flow">
  <div><b>Coarse precipitation</b><small>How much rain?</small></div><span>+</span>
  <div><b>Terrain + wind + boundary layer</b><small>Where should it fall?</small></div><span>→</span>
  <div><b>Local wet/dry weights</b><small>Ridges, slopes, valleys</small></div><span>→</span>
  <div><b>Fine precipitation</b><small>Same coarse-cell mean</small></div>
</div>

This is best understood as **physical spatial redistribution**, not as a learned forecast model. It does not discover relationships from observations, and it cannot correct a climate model that predicts the wrong regional rainfall total.

## What the papers contribute

The method is described most directly by Karger and colleagues in the [2021 CHELSA-W5E5 methods paper](https://www.nature.com/articles/s41597-021-01084-6). That paper combines daily atmospheric forcing with high-resolution terrain to produce a global 1 km data set. For precipitation it defines the windward/leeward effect, a boundary-layer correction, valley exposure, and a final coarse-cell normalization.

The [2023 CHELSA-W5E5 paper](https://essd.copernicus.org/articles/15/2445/2023/) presents the resulting 1979–2016 global data set, evaluates it against station and gridded products, and emphasizes applications that need kilometre-scale climate forcing. The important distinction for our work is:

- the papers describe a **daily, global, approximately 1 km** production system;
- DeepScale uses the same published precipitation equations as a reusable component for **seasonal regional forecasts**, often on a coarser target grid;
- therefore our implementation is a faithful equation-level reproduction, but our seasonal example is an **adaptation**, not an exact recreation of the published global product.

The papers report broad advantages from resolving topography, particularly in mountainous regions. That supports trying the method; it does not guarantee improved forecast accuracy in every region, season, resolution, or observational reference.

## What we reproduced in DeepScale

We translated the published precipitation path into xarray-based Python and exposed it as `method="chelsa"`. The implementation:

- computes a SAGA-style windward/leeward terrain index from elevation and wind;
- applies the paper's boundary-layer equation, including the published **500 m offset** by default;
- applies the CHELSA exposure correction when that field is supplied;
- multiplies these pieces to form the paper's local precipitation weight;
- normalizes within every parent grid cell so the coarse precipitation mean is conserved;
- learns atmospheric climatologies from training years only, preventing held-out years from leaking into cross-validation.

The 500 m value is a documented model constant from the published equation, but DeepScale makes it a parameter (`pbl_offset_m`) so sensitivity tests are possible. The default is the paper value.

### How we checked the reproduction

We used several complementary checks rather than treating a visually plausible map as proof:

| Check | What it tells us | Result |
|---|---|---|
| Equation-level unit tests | Published transformations behave as specified | Passed |
| Conservation tests | Fine-grid output preserves each coarse-cell mean | Errors at floating-point precision (~10⁻¹⁴) |
| SAGA wind-effect comparison | Our terrain effect agrees with an independent implementation | Central-domain maximum difference 0.00485; edge effects larger |
| Cross-validation safeguards | Held-out atmospheric data cannot enter fitting | Enforced and tested |
| Repository test suite | Integration did not break other DeepScale behavior | 977 passed, 56 skipped |
| Real-data example | The complete data and plotting path runs | Rwanda MAM 2015 completed successfully |

The SAGA discrepancy is concentrated at artificial domain edges, where a directional terrain algorithm has less surrounding information. Regional analyses should therefore include a buffer and crop it away after calculation.

We intentionally followed the **published equations** where the archived workflow and paper were ambiguous. In particular, the legacy script appears to overwrite an intermediate boundary-layer correction. DeepScale retains the multiplicative form stated in the paper. This is documented and tested, but it is also why we describe the result as paper-defined rather than bit-for-bit identical to every historical CHELSA script.

## What worked well—and what did not

### Strongest positive result: high terrain in Rwanda

In a 24-year placement test for Rwanda's MAM season, restricted to observation cells at or above 2,500 m, CHELSA achieved an RMSE of **0.549 mm/day**, compared with **0.867** for bilinear interpolation and **1.077** for uniform redistribution. It beat bilinear interpolation in 23 of 24 years. A bootstrap interval for the CHELSA-minus-bilinear RMSE difference was −0.390 to −0.241 mm/day.

That is the use case the method is designed for: resolving precipitation placement in steep terrain. It also produced physically interpretable ridge/valley patterns and preserved the driving model's coarse rainfall totals exactly.

### The broader result was negative

Across national domains, CHELSA did not beat simple alternatives:

| Region and season | Uniform | Bilinear | Observed climatology | CHELSA |
|---|---:|---:|---:|---:|
| Rwanda MAM | 0.558 | 0.515 | **0.345** | 0.641 |
| Rwanda OND | 0.663 | 0.581 | **0.357** | 0.783 |
| Ethiopia MAM | 0.531 | 0.427 | **0.275** | 0.560 |
| Ethiopia OND | 0.351 | 0.285 | **0.196** | 0.371 |

*RMSE in mm/day; lower is better. These are precipitation-placement experiments, not a complete assessment of seasonal forecast skill.*

The likely explanation is not that terrain is irrelevant. Rather, a fixed physical redistribution cannot fully represent convection, lake effects, regional circulation, observation error, or season-specific rainfall climatology. At the resolutions tested, those effects can dominate the added orographic signal. Observed climatology was consistently strongest, which shows the value of learning a stable local rainfall pattern when historical observations are available.

## Did someone already implement this?

Yes—but we did not find a drop-in implementation suitable for DeepScale.

| Project | What is available | Reuse assessment |
|---|---|---|
| [Official CHELSA-W5E5 source](https://zenodo.org/records/8010301) | The authors' full workflow; also linked to the [versioned GitHub repository](https://github.com/greenmind1980/CHELSA-W5E5/tree/V1.0) | Essential reference, but a legacy Python 2.7/SAGA/GDAL pipeline under GPL-3.0—not a modern Python library we can copy into MIT-licensed DeepScale |
| [CHELSA EarthEnv](https://gitlabext.wsl.ch/karger/chelsa_earthenv) | Earlier global workflow, including additional cloud refinement | Valuable comparison and validation material; still a production pipeline rather than a reusable seasonal downscaling component |
| [SAGA GIS wind effect](https://github.com/saga-gis/saga-gis/blob/master/saga-gis/src/tools/terrain_analysis/ta_morphometry/wind_effect.cpp) | Open implementation of the terrain wind-effect component | Useful independent numerical reference; covers only one part of CHELSA and is GPL-licensed |
| [CHELSA-CMIP6](https://www.chelsa-climate.org/tutorials) | Official Python tooling for downscaled CMIP6 products | Uses an anomaly/delta workflow with a CHELSA climatology; not the same mechanistic precipitation kernel |
| Rchelsa / ClimDatDownloadR | Download, crop, and access published CHELSA products | Data-access tools, not implementations of the algorithm |

Our conclusion is that we are **not inventing the science from scratch**: the equations, official scripts, and an independent SAGA component all exist and informed our tests. However, no permissively licensed, maintained, xarray-native implementation of the exact precipitation redistribution kernel was identified. An independent implementation is therefore reasonable for DeepScale, provided we continue to cite the papers, preserve provenance, and test against independent references rather than copying GPL code.

## Recommendation

Keep CHELSA in DeepScale as an **optional, experimental precipitation downscaler** with the following operating guidance:

1. Use it when terrain-driven placement matters and wind, boundary-layer, elevation, and preferably exposure inputs are available.
2. Always compare it against bilinear interpolation and an observed-climatology baseline under honest multi-year validation.
3. Prefer it only where it demonstrates local value—high-elevation Rwanda is encouraging; national East Africa results are not.
4. Use buffered domains to reduce terrain-index edge artifacts.
5. Do not describe the current seasonal implementation as a reproduction of the complete 1 km daily CHELSA-W5E5 product.

The next useful evidence would be an independent high-resolution precipitation reference, more mountainous regions, daily inputs closer to the paper, and sensitivity tests for atmospheric resolution and the 500 m boundary-layer offset. Until then, the right status is **implemented and numerically verified, scientifically promising in a narrow regime, but not generally superior**.

---

### Sources and reproducibility

Primary papers: [Karger et al. (2021), *Scientific Data*](https://www.nature.com/articles/s41597-021-01084-6); [Karger et al. (2023), *Earth System Science Data*](https://essd.copernicus.org/articles/15/2445/2023/). Model background: [CHELSA v2.0 overview](https://www.chelsa-climate.org/models/chelsa_v2.0). Software audit links appear above.

DeepScale materials: `src/deepscale/methods/chelsa.py`; `tests/test_chelsa.py`; `validation/chelsa/`; `examples/demo_chelsa.py`. Validation figures and values in this brief were produced during the East Africa CHELSA investigation and are retained with the project artifacts.
