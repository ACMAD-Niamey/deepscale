"""Generalized ROC (GROC) — multi-category discrimination score.

A single-number summary of how well a tercile-probability forecast
discriminates between the three tercile categories. Implemented as a thin
wrapper over scikit-learn's :func:`roc_auc_score` with
``multi_class="ovo"`` (Hand & Till 2001), which is mathematically
equivalent to the Mason & Weigel 2011 Generalized Discrimination Score
for ensemble forecasts on 3-tercile probabilistic input.

Range ``[0, 1]``; ``0.5`` = no discrimination, ``1.0`` = perfect.

Input contract: ``forecast`` must have a ``tercile`` dim of size 3
holding the probabilities for (below-normal, normal, above-normal).
``obs`` is continuous and is converted to tercile labels internally
using :func:`deepscale.metrics.rpss._cpt_boundaries`.
"""

import warnings

import numpy as np
import xarray as xr
from sklearn.metrics import roc_auc_score

from .base import MetricBase
from ..registry import register_metric
from .rpss import _cpt_boundaries


def _obs_to_categories(obs_vals, *, loo_boundaries=False):
    """Categorize per-gridpoint observations into terciles {0, 1, 2}.

    Returns an int array shaped like ``obs_vals`` (n_year, ...). NaN cells
    (whether from missing obs or undefined boundaries) get the sentinel
    value ``-1``; downstream consumers should drop these.
    """
    if loo_boundaries:
        n = obs_vals.shape[0]
        obs_cat = np.full(obs_vals.shape, -1, dtype=int)
        for idx in range(n):
            mask = np.arange(n) != idx
            t33, t67 = _cpt_boundaries(obs_vals[mask])
            obs_yr = obs_vals[idx]
            cats = np.where(t33 > obs_yr, 0, np.where(t67 > obs_yr, 1, 2))
            nan_mask = np.isnan(obs_yr) | np.isnan(t33)
            cats[nan_mask] = -1
            obs_cat[idx] = cats
        return obs_cat

    t33, t67 = _cpt_boundaries(obs_vals)
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2)).astype(int)
    nan_mask = np.isnan(obs_vals) | np.isnan(t33)
    obs_cat[nan_mask] = -1
    return obs_cat


def _groc_from_flat(y_true_flat, y_score_flat):
    """Compute GROC for a flat (n_samples,) label vector and
    (n_samples, 3) probability matrix. Drops invalid rows; returns
    ``float("nan")`` if fewer than two distinct categories remain.
    """
    valid = (y_true_flat >= 0) & ~np.isnan(y_score_flat).any(axis=1)
    yt = y_true_flat[valid]
    ys = y_score_flat[valid]
    classes = np.unique(yt)
    if yt.size == 0 or classes.size < 2:
        return float("nan")
    if classes.size == 2:
        # Only two obs categories present (common per-cell on dry-masked
        # seasonal domains, e.g. below/above with no normal year in a short
        # record). sklearn's multi_class="ovo" requires every y_score column's
        # class in y_true and raises otherwise, so compute the single
        # Hand-&-Till pair directly — the average of the two directed binary
        # AUCs, exactly what sklearn's ovo evaluates for one pair.
        lo, hi = int(classes[0]), int(classes[1])
        a_hi = roc_auc_score((yt == hi).astype(int), ys[:, hi])
        a_lo = roc_auc_score((yt == lo).astype(int), ys[:, lo])
        return float(0.5 * (a_hi + a_lo))
    return float(roc_auc_score(yt, ys, multi_class="ovo", average="macro"))


@register_metric("generalized_roc", aliases=("groc",))
class GeneralizedROCMetric(MetricBase):
    """Multi-category discrimination score for tercile-probability forecasts.

    Range ``[0, 1]``; ``0.5`` = no discrimination, ``1.0`` = perfect.
    Returns NaN with a warning when the input doesn't contain enough
    distinct obs categories to compute a discrimination score.
    """

    def compute(self, forecast, obs, *, spatial=False, loo_boundaries=False, **kwargs):
        if "tercile" not in forecast.dims or forecast.sizes["tercile"] != 3:
            raise ValueError(
                "generalized_roc requires a 'tercile' dim of size 3; "
                f"got dims={tuple(forecast.dims)}"
            )

        # Forecast and obs are paired by flattening their non-class axes in the
        # same order. Drive the order off obs.dims so a forecast with permuted
        # spatial dims still pairs correctly.
        obs_t = obs.transpose("year", ...)
        obs_cat = _obs_to_categories(obs_t.values, loo_boundaries=loo_boundaries)
        fcst_t = forecast.transpose(*obs_t.dims, "tercile")
        fcst_vals = fcst_t.values

        if spatial:
            spatial_dims = [d for d in obs_t.dims if d != "year"]
            spatial_shape = tuple(obs_t.sizes[d] for d in spatial_dims)
            result = np.full(spatial_shape, np.nan)
            for idx in np.ndindex(spatial_shape):
                y_true = obs_cat[(slice(None),) + idx]
                y_score = fcst_vals[(slice(None),) + idx]
                result[idx] = _groc_from_flat(y_true, y_score)
            coords = {d: obs[d] for d in spatial_dims}
            return xr.DataArray(result, dims=spatial_dims, coords=coords)

        y_true_flat = obs_cat.flatten()
        y_score_flat = fcst_vals.reshape(-1, 3)
        score = _groc_from_flat(y_true_flat, y_score_flat)
        if np.isnan(score):
            warnings.warn(
                "generalized_roc: fewer than two distinct obs categories after "
                "NaN-masking; returning NaN",
                RuntimeWarning,
                stacklevel=2,
            )
        return score
