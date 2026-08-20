"""Reusable terrain covariates built from Copernicus GLO-90.

`load_terrain(like)` downloads Copernicus DEM GLO-90 elevation tiles (ESA /
Airbus, served without credentials from the public AWS bucket
``s3://copernicus-dem-90m``), block-averages them onto the (lat, lon) grid of
`like`, and returns a Dataset with cell-mean ``elevation`` and sub-grid
``roughness``. Results are cached as a small NetCDF so only the first use of a
given grid needs the network.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

__all__ = ["load_terrain"]

# One Cloud-Optimized GeoTIFF per 1x1 degree tile (~4 MB); ocean tiles simply
# do not exist (404). Each tile is downloaded whole with plain HTTP and read
# from a temporary file: range-reading the COGs over GDAL/vsicurl proved
# pathologically slow against this bucket (>30 min for a few hundred tiles),
# while full parallel downloads finish the same region in minutes.
_DEM_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM/"
    "Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM.tif"
)
_HTTP_TIMEOUT = 300  # seconds per tile download
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deepscale"


def _tile_urls(region):
    """URLs of every 1x1 degree GLO-90 tile touching [south, north, west, east]."""
    south, north, west, east = region
    urls = []
    for lat in range(int(np.floor(south)), int(np.ceil(north))):
        for lon in range(int(np.floor(west)), int(np.ceil(east))):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            urls.append(_DEM_URL.format(ns=ns, ew=ew))
    return urls


def _cell_edges(centers):
    """Cell edges for a (possibly descending) 1-D coordinate, ascending.

    Interior edges are midpoints between neighboring centers; the outer edges
    extrapolate the first/last spacing. Identical to a half-step expansion for
    a regular grid, and well-behaved for a slightly irregular one.
    """
    c = np.sort(np.asarray(centers, dtype=float))
    if c.size < 2:
        raise ValueError(
            "load_terrain needs at least 2 points along each of lat/lon to "
            "derive cell edges from the grid"
        )
    inner = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (c[1] - c[0]) / 2.0
    last = c[-1] + (c[-1] - c[-2]) / 2.0
    return np.concatenate([[first], inner, [last]])


def _read_tile(url):
    """Download one GLO-90 tile and read it -> (x, y, values) or None.

    None means the tile does not exist (ocean, 404) or the bucket is
    unreachable; the caller decides whether zero successful tiles is an error.

    The tile bytes go to a temporary file that is removed as soon as the
    values are in memory — the aggregate NetCDF written by `load_terrain` is
    the only thing this module persists. The read uses rasterio directly
    (no rioxarray/pyproj CRS objects): this function runs on worker threads,
    and constructing CRS objects there produced interpreter-shutdown noise.
    """
    import tempfile

    import rasterio
    import requests

    try:
        r = requests.get(url, timeout=_HTTP_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with tempfile.TemporaryDirectory(prefix="deepscale-glo90-") as tmp:
            path = Path(tmp) / url.rsplit("/", 1)[-1]
            path.write_bytes(r.content)
            with rasterio.open(path) as src:
                band = src.read(1, masked=True)
                tr = src.transform
                ny, nx = band.shape
                x = tr.c + tr.a * (np.arange(nx) + 0.5)
                y = tr.f + tr.e * (np.arange(ny) + 0.5)
                vals = np.ma.filled(band.astype("float64"), np.nan)
            return x, y, vals
    except Exception:
        return None


def _grid_coords(like):
    for name in ("lat", "lon"):
        if name not in like.coords:
            raise ValueError(
                f"`like` must carry 1-D 'lat' and 'lon' coordinates; missing {name!r}"
            )
    lat = np.asarray(like["lat"].values, dtype=float)
    lon = np.asarray(like["lon"].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("`like` must carry 1-D 'lat' and 'lon' coordinates")
    return lat, lon


def _cache_path(cache_dir, lat, lon):
    digest = hashlib.sha1(
        np.round(lat, 6).tobytes() + b"|" + np.round(lon, 6).tobytes()
    ).hexdigest()[:10]
    return Path(cache_dir) / f"terrain_glo90_{lat.size}x{lon.size}_{digest}.nc"


def _matches(cached, lat, lon):
    return (
        {"elevation", "roughness"} <= set(cached.data_vars)
        and cached["lat"].size == lat.size
        and cached["lon"].size == lon.size
        and np.allclose(cached["lat"].values, lat)
        and np.allclose(cached["lon"].values, lon)
    )


def load_terrain(like, cache_dir=None):
    """Copernicus GLO-90 elevation + roughness on the grid of `like`.

    Builds terrain covariates with one call: real elevation is
    *block-averaged* onto each grid cell
    (a sampled DEM point would be the height of one hillside, not of a
    multi-km cell), and the sub-grid standard deviation is kept as a
    roughness covariate.

    Data provenance: Copernicus DEM GLO-90 (ESA / Airbus), downloaded tile by
    tile from the public AWS bucket ``s3://copernicus-dem-90m`` (no
    credentials; ocean tiles do not exist and are skipped). Full-resolution
    (~90 m) tiles are fetched in parallel to temporary files, read with
    rioxarray, and discarded once aggregated — nothing but the NetCDF result
    is kept on disk.

    The result is cached as a small NetCDF, so the **first use of a given grid
    needs network access**; every later call with a matching grid is served
    fully offline from the cache. A cached file whose grid no longer matches
    `like` is rebuilt.

    Parameters
    ----------
    like : xr.DataArray or xr.Dataset
        Supplies the target grid: 1-D ``lat`` and ``lon`` coordinates (either
        orientation, at least 2 points each). The fetch region is derived
        from the grid's cell edges.
    cache_dir : path-like, optional
        Directory for the NetCDF cache. Default: ``~/.cache/deepscale``.

    Returns
    -------
    xr.Dataset
        ``elevation`` (cell-mean, m) and ``roughness`` (sub-grid standard
        deviation, m), dims ``(lat, lon)`` on exactly `like`'s coordinates.

    Raises
    ------
    RuntimeError
        If no tile could be fetched and no cache exists (offline with a cold
        cache, or an all-ocean region).
    ValueError
        If `like` lacks 1-D lat/lon coordinates or has fewer than 2 points
        along either.
    """
    lat_v, lon_v = _grid_coords(like)
    lat_edges = _cell_edges(lat_v)
    lon_edges = _cell_edges(lon_v)
    cache = _cache_path(cache_dir or _DEFAULT_CACHE_DIR, lat_v, lon_v)

    if cache.exists():
        with xr.open_dataset(cache) as cached:
            if _matches(cached, lat_v, lon_v):
                return cached.load().assign_coords(lat=like["lat"], lon=like["lon"])
        # grid changed (or the file is not ours): fall through and rebuild

    lat_sorted, lon_sorted = np.sort(lat_v), np.sort(lon_v)
    ny, nx = lat_v.size, lon_v.size
    tot = np.zeros((ny, nx))
    tot_sq = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))

    region = [lat_edges[0], lat_edges[-1], lon_edges[0], lon_edges[-1]]
    urls = _tile_urls(region)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=min(16, len(urls))) as pool:
        for tile in pool.map(lambda u: _read_tile(u), urls):
            if tile is None:
                continue
            n_ok += 1
            x, y, vals = tile
            good = np.isfinite(vals)
            lons = np.broadcast_to(x[None, :], vals.shape)[good]
            lats = np.broadcast_to(y[:, None], vals.shape)[good]
            v = vals[good]
            iy = np.digitize(lats, lat_edges) - 1
            ix = np.digitize(lons, lon_edges) - 1
            keep = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
            iy, ix, v = iy[keep], ix[keep], v[keep]
            np.add.at(tot, (iy, ix), v)
            np.add.at(tot_sq, (iy, ix), v**2)
            np.add.at(cnt, (iy, ix), 1.0)

    if n_ok == 0:
        raise RuntimeError(
            "load_terrain could not fetch any Copernicus GLO-90 elevation tile "
            f"for region [S,N,W,E]={np.round(region, 3).tolist()} and no cache "
            f"exists at {cache}. The first use of a grid needs network access "
            "to the public bucket s3://copernicus-dem-90m (via "
            "copernicus-dem-90m.s3.amazonaws.com); later calls are served "
            "offline from the cache. If the region is truly all ocean there "
            "is no terrain to load. Alternatively, build the `terrain=` input "
            "yourself: an xr.Dataset on the obs fine grid with an elevation "
            "variable named one of 'elevation'/'dem'/'z' plus optional extra "
            "numeric (lat, lon) covariates."
        )

    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 0, tot_sq / np.maximum(cnt, 1) - mean**2, np.nan)
    # The accumulation grid is ascending; flip onto `like`'s coordinate order
    # if it differs.
    if not np.array_equal(lat_v, lat_sorted):
        mean, var = mean[::-1], var[::-1]
    if not np.array_equal(lon_v, lon_sorted):
        mean, var = mean[:, ::-1], var[:, ::-1]

    terrain = xr.Dataset(
        {
            "elevation": (("lat", "lon"), mean),
            "roughness": (("lat", "lon"), np.sqrt(np.maximum(var, 0.0))),
        },
        coords={"lat": like["lat"].values, "lon": like["lon"].values},
        attrs={
            "source": (
                "Copernicus DEM GLO-90 (ESA/Airbus; AWS s3://copernicus-dem-90m), "
                "full-resolution 90 m tiles block-averaged onto the target grid"
            ),
            "tiles_retrieved": f"{n_ok}/{len(urls)} (missing tiles are ocean)",
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    terrain.to_netcdf(cache)
    return terrain.assign_coords(lat=like["lat"], lon=like["lon"])
