"""CHELSA-style precipitation redistribution from published equations.

This is an independent implementation of the precipitation kernel described
by Karger et al. (2021, 2023).  It does not copy the GPL CHELSA/SAGA source.
The production CHELSA workflow also uses multilevel B-splines, a 3 km wind
grid, a 30 arc-second DEM, and a precomputed high-elevation exposure surface.
Those data-engineering steps are intentionally kept outside this numerical
kernel.

References
----------
Karger et al. (2021), Scientific Data 8, 307.
https://doi.org/10.1038/s41597-021-01084-6

Karger et al. (2023), Earth System Science Data 15, 2445-2464.
https://doi.org/10.5194/essd-15-2445-2023
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

from .base import MethodBase
from ..registry import register_method

EARTH_RADIUS_M = 6_371_008.8


def _spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Median north/east grid spacing in metres for a regular lat/lon grid."""
    dy = np.deg2rad(np.median(np.abs(np.diff(lat)))) * EARTH_RADIUS_M
    dx = (np.deg2rad(np.median(np.abs(np.diff(lon)))) * EARTH_RADIUS_M
          * np.cos(np.deg2rad(float(np.mean(lat)))))
    if (
        lat.ndim != 1
        or lon.ndim != 1
        or lat.size < 2
        or lon.size < 2
        or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0))
        or not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0))
        or not np.isfinite(dx + dy)
        or min(dx, dy) <= 0
    ):
        raise ValueError("elevation needs monotonic lat/lon coordinates")
    return float(dy), float(dx)


def _ray_terms(
    z: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    *,
    dy_m: float,
    dx_m: float,
    max_distance_km: float,
    acceleration: float,
    sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Short- and long-distance terrain angles along one wind trajectory.

    Distances and transformations follow the equations used by CHELSA's SAGA
    wind-effect operator. ``sign=-1`` samples upwind; ``sign=+1`` downwind.
    """
    rows, cols = np.indices(z.shape, dtype=float)
    rr = rows.copy()
    cc = cols.copy()
    short_sum = np.zeros(z.shape, dtype=float)
    short_w = np.zeros(z.shape, dtype=float)
    long_sum = np.zeros(z.shape, dtype=float)
    long_w = np.zeros(z.shape, dtype=float)

    step = min(dx_m, dy_m)
    distance = step
    move = step
    while distance <= max_distance_km * 1000.0:
        # SAGA samples the local direction at the current ray position before
        # taking each accelerating step; this allows curved wind trajectories.
        ray_east = map_coordinates(east, [rr, cc], order=1, mode="constant", cval=np.nan)
        ray_north = map_coordinates(north, [rr, cc], order=1, mode="constant", cval=np.nan)
        rr = rr + sign * move * ray_north / dy_m
        cc = cc + sign * move * ray_east / dx_m
        sampled = map_coordinates(z, [rr, cc], order=1, mode="constant", cval=np.nan)
        valid = np.isfinite(z) & np.isfinite(sampled)
        angle = np.arctan2(z - sampled, np.sqrt(distance))
        ws = move / distance
        wl = move / np.log1p(distance)
        short_sum[valid] += ws * angle[valid]
        short_w[valid] += ws
        long_sum[valid] += wl * angle[valid]
        long_w[valid] += wl
        move *= acceleration
        distance += move

    short = np.divide(short_sum, short_w, out=np.zeros_like(z), where=short_w > 0)
    long = np.divide(long_sum, long_w, out=np.zeros_like(z), where=long_w > 0)
    return short, long


def _positive_transform(value: np.ndarray, *, root: float) -> np.ndarray:
    transformed = np.empty_like(value)
    positive = value > 0
    transformed[positive] = 1.0 + np.log1p(value[positive])
    transformed[~positive] = 1.0 / (1.0 + np.log1p(-value[~positive]))
    return np.maximum(transformed, np.finfo(float).tiny) ** root


def wind_effect(
    elevation: xr.DataArray,
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    *,
    max_distance_km: float = 300.0,
    acceleration: float = 1.5,
) -> xr.DataArray:
    """Dimensionless CHELSA/SAGA-style windward-leeward index ``H``.

    ``u_wind`` and ``v_wind`` are eastward and northward components on, or
    interpolatable to, the elevation grid. Wind speed does not scale the
    result; its direction determines the terrain trajectories.
    """
    elevation = elevation.transpose("lat", "lon")
    u = u_wind.interp_like(elevation).transpose("lat", "lon")
    v = v_wind.interp_like(elevation).transpose("lat", "lon")
    speed = np.hypot(u.values, v.values)
    east = np.divide(u.values, speed, out=np.zeros_like(speed), where=speed > 0)
    north = np.divide(v.values, speed, out=np.zeros_like(speed), where=speed > 0)
    dy_m, dx_m = _spacing_m(elevation.lat.values, elevation.lon.values)
    z = elevation.values.astype(float)

    # SAGA 7.6's variable-direction implementation calls both Get_Luv and
    # Get_Lee with reverse tracing. Reusing the same ray terms here follows
    # the actual production dependency, including that non-obvious behavior.
    luv, lee = _ray_terms(z, east, north, dy_m=dy_m, dx_m=dx_m,
                          max_distance_km=max_distance_km,
                          acceleration=acceleration)
    effect = (_positive_transform(2.0 * luv, root=0.25)
              * _positive_transform(lee, root=0.125))
    effect[~np.isfinite(z) | ~np.isfinite(speed) | (speed <= 0)] = np.nan
    return xr.DataArray(effect, dims=("lat", "lon"), coords=elevation.coords,
                        name="wind_effect",
                        attrs={"long_name": "CHELSA windward-leeward index"})


def _parent_indices(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    return np.abs(fine[:, None] - coarse[None, :]).argmin(axis=1)


def _normalize_by_parent(intensity: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """Normalize every coarse parent cell to mean intensity one."""
    nx = int(ix.max()) + 1
    labels = iy[:, None] * nx + ix[None, :]
    valid = np.isfinite(intensity)
    size = (int(iy.max()) + 1) * nx
    sums = np.bincount(labels[valid], weights=intensity[valid], minlength=size)
    counts = np.bincount(labels[valid], minlength=size)
    means = np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)
    parent_means = means[labels]
    return np.divide(
        intensity, parent_means, out=np.full_like(intensity, np.nan, dtype=float),
        where=valid & np.isfinite(parent_means) & (parent_means > 0),
    )


def chelsa_precipitation(
    coarse_precip: xr.DataArray,
    elevation: xr.DataArray,
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    *,
    boundary_layer_height: xr.DataArray | None = None,
    coarse_orography: xr.DataArray | None = None,
    exposure: xr.DataArray | None = None,
    pbl_offset_m: float = 500.0,
    height_scale_m: float = 9000.0,
    max_distance_km: float = 300.0,
) -> xr.DataArray:
    """Redistribute coarse precipitation using CHELSA orographic intensity.

    Implements Karger et al. (2021) equations for ``p_I = E * H_B`` and
    ``p_fine = p_I / mean_parent(p_I) * p_coarse``. Consequently, the mean of
    every fine group exactly equals its parent coarse value (apart from
    floating-point error). PBL correction is applied when both PBL height and
    coarse-grid orography are provided; ``exposure`` defaults to one.
    """
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")
    if height_scale_m <= 0:
        raise ValueError("height_scale_m must be positive")
    if bool((coarse_precip < 0).any()):
        raise ValueError("CHELSA precipitation input must be non-negative")

    fine = elevation.transpose("lat", "lon")
    coarse = coarse_precip.transpose(..., "lat", "lon")
    h = wind_effect(fine, u_wind, v_wind,
                    max_distance_km=max_distance_km).values
    iy = _parent_indices(fine.lat.values, coarse.lat.values)
    ix = _parent_indices(fine.lon.values, coarse.lon.values)

    if boundary_layer_height is not None or coarse_orography is not None:
        if boundary_layer_height is None or coarse_orography is None:
            raise ValueError("PBL correction needs boundary_layer_height and coarse_orography")
        pbl = boundary_layer_height.interp_like(fine).values
        zc = coarse_orography.interp_like(fine).values
        pbl_z = pbl + zc + pbl_offset_m
        distance = np.abs(fine.values - pbl_z)
        nx = int(ix.max()) + 1
        labels = iy[:, None] * nx + ix[None, :]
        size = (int(iy.max()) + 1) * nx
        valid = np.isfinite(distance)
        maxima = np.full(size, -np.inf)
        np.maximum.at(maxima, labels[valid], distance[valid])
        maxima[~np.isfinite(maxima)] = np.nan
        maximum = maxima[labels]
        # Karger et al. (2021): H_B = H / (1 - (distance - z_max) / h).
        # Since z_max is the parent-cell maximum distance, the denominator is
        # >= 1.  This deliberately follows the published equation rather than
        # the archived W5E5 driver's different intermediate-grid operation.
        denominator = 1.0 - (distance - maximum) / height_scale_m
        h = h / denominator

    e = 1.0 if exposure is None else exposure.interp_like(fine).values
    weights = _normalize_by_parent(np.asarray(e) * h, iy, ix)
    coarse_at_fine = coarse.isel(
        lat=xr.DataArray(iy, dims="lat"),
        lon=xr.DataArray(ix, dims="lon"),
    ).assign_coords(lat=fine.lat, lon=fine.lon)
    result = coarse_at_fine * xr.DataArray(weights, dims=("lat", "lon"),
                                            coords=fine.coords)
    result.name = "precip"
    result.attrs.update(coarse_precip.attrs)
    result.attrs["downscaling"] = "CHELSA orographic redistribution"
    result.attrs["chelsa_variant"] = "paper-defined Karger 2021/2023"
    return result


def _training_climatology(field: xr.DataArray, years: xr.DataArray) -> xr.DataArray:
    """Select training years when present, then average non-spatial axes."""
    if "year" in field.dims:
        field = field.sel(year=years)
    elif "time" in field.dims:
        try:
            keep = field.time.dt.year.isin(years)
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                "CHELSA time-bearing atmospheric inputs need datetime-like "
                "coordinates so training years can be selected without leakage"
            ) from exc
        field = field.sel(time=keep)
        if field.sizes["time"] == 0:
            raise ValueError("CHELSA atmospheric inputs do not cover the training years")
    reduce = [dim for dim in field.dims if dim not in ("lat", "lon")]
    return field.mean(reduce) if reduce else field


@register_method("chelsa")
class CHELSAMethod(MethodBase):
    """Paper-defined CHELSA precipitation redistribution.

    Training stores fine terrain/static exposure and training-period
    climatologies of wind and PBL inputs.  Prediction may instead receive a
    two-dimensional ``forecast_atmosphere`` Dataset with ``u_wind``,
    ``v_wind``, and optional ``boundary_layer_height`` / ``coarse_orography``.
    Requiring forecast auxiliaries to be 2-D prevents accidental held-year
    leakage in DeepScale CV loops.
    """

    def fit(self, hindcast, obs, **kwargs):
        required = ("terrain", "u_wind", "v_wind")
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise TypeError(
                "CHELSA fit requires " + ", ".join(f"`{name}=`" for name in missing)
            )
        terrain = kwargs["terrain"]
        if isinstance(terrain, xr.Dataset):
            if "elevation" not in terrain:
                raise ValueError("terrain Dataset requires an `elevation` variable")
            elevation = terrain["elevation"]
        else:
            elevation = terrain
        self.elevation_ = elevation.interp(lat=obs.lat, lon=obs.lon)
        self.u_wind_ = _training_climatology(kwargs["u_wind"], hindcast.year)
        self.v_wind_ = _training_climatology(kwargs["v_wind"], hindcast.year)
        pbl = kwargs.get("boundary_layer_height")
        orography = kwargs.get("coarse_orography")
        if (pbl is None) != (orography is None):
            raise ValueError(
                "CHELSA PBL correction needs both `boundary_layer_height=` "
                "and `coarse_orography=`"
            )
        self.boundary_layer_height_ = (
            None if pbl is None else _training_climatology(pbl, hindcast.year)
        )
        self.coarse_orography_ = (
            None if orography is None else _training_climatology(orography, hindcast.year)
        )
        exposure = kwargs.get("exposure")
        self.exposure_ = None if exposure is None else exposure.interp_like(self.elevation_)
        self.pbl_offset_m_ = float(kwargs.get("pbl_offset_m", 500.0))
        self.height_scale_m_ = float(kwargs.get("height_scale_m", 9000.0))
        self.max_distance_km_ = float(kwargs.get("max_distance_km", 300.0))

    def predict(self, forecast, **kwargs):
        atmosphere = kwargs.get("forecast_atmosphere")
        if atmosphere is None:
            u = self.u_wind_
            v = self.v_wind_
            pbl = self.boundary_layer_height_
            orography = self.coarse_orography_
        else:
            if any(dim not in ("lat", "lon") for dim in atmosphere.dims):
                raise ValueError(
                    "forecast_atmosphere must contain only 2-D lat/lon fields; "
                    "select the forecast period before calling predict()"
                )
            for name in ("u_wind", "v_wind"):
                if name not in atmosphere:
                    raise ValueError(f"forecast_atmosphere requires `{name}`")
            u, v = atmosphere["u_wind"], atmosphere["v_wind"]
            pbl = atmosphere.get("boundary_layer_height")
            orography = atmosphere.get("coarse_orography")

        result = chelsa_precipitation(
            forecast,
            self.elevation_,
            u,
            v,
            boundary_layer_height=pbl,
            coarse_orography=orography,
            exposure=self.exposure_,
            pbl_offset_m=self.pbl_offset_m_,
            height_scale_m=self.height_scale_m_,
            max_distance_km=self.max_distance_km_,
        )
        leading = [dim for dim in forecast.dims if dim not in ("lat", "lon")]
        return result.transpose(*leading, "lat", "lon")
