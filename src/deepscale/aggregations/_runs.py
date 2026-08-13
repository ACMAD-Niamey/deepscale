"""Rolling totals and consecutive-run lengths over a daily step axis.

Onset's false-start guard, cessation, and dry-spell statistics are one question
read three ways: where are the runs of consecutive days at or below a
threshold. This module answers it once, for every cell, year and member
simultaneously. Nothing here knows what a rainy season is.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["rolling_total", "dry_run_lengths", "has_data"]


def rolling_total(rain: xr.DataArray, window: int, *, dim: str = "step") -> xr.DataArray:
    """Total over ``window`` steps *starting* at each position.

    ``xarray.rolling`` reports the window *ending* at each position, so the
    result is shifted back by ``window - 1``. A position whose window runs off
    the end of the axis, or contains a NaN, is NaN: an accumulation cannot be
    confirmed from incomplete data.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    rolled = rain.rolling({dim: window}, min_periods=window).sum()
    return rolled.shift({dim: -(window - 1)})


def _run_lengths_1d(dry: np.ndarray) -> np.ndarray:
    """Length of the run of True ending at each position; 0 where False.

    Walks the axis once. ``maximum.accumulate`` over the indices of False
    entries gives, at each position, the index of the most recent False, and
    the distance to it is the run length.
    """
    n = dry.shape[-1]
    idx = np.arange(n)
    reset = np.where(~dry, idx, -1)
    last_false = np.maximum.accumulate(reset, axis=-1)
    return np.where(dry, idx - last_false, 0).astype(np.int64)


def dry_run_lengths(rain: xr.DataArray, threshold: float, *,
                    dim: str = "step") -> xr.DataArray:
    """Length of the consecutive at-or-below-``threshold`` run ending at each step.

    Wet steps are 0. The threshold is inclusive, matching the "1 mm/day or
    less" wording of the criterion.

    NaN steps count as not dry, because ``NaN <= threshold`` is False. A gap in
    the record therefore breaks a run rather than extending it, which is the
    conservative direction: missing data never manufactures a dry spell.
    """
    dry = rain <= threshold
    return xr.apply_ufunc(
        _run_lengths_1d,
        dry,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[np.int64],
    ).transpose(*rain.dims)


def has_data(rain: xr.DataArray, *, dim: str = "step") -> xr.DataArray:
    """True where the cell has at least one finite value along ``dim``.

    The three-state ``occurred`` contract depends on this. Rolling sums over an
    all-NaN cell yield NaN, NaN fails every comparison, and a naive ``any``
    would report the cell as a failed season rather than as absent data.
    """
    return rain.notnull().any(dim)
