"""CCA downscaling — SVD-based, matching CPT Fortran 17.8.3 (cca.F95).

Algorithm (CPT's full_cca + cca_predict):
  1. Center data (remove mean), apply sqrt(cos(lat)) weighting
  2. SVD of weighted anomalies → EOF patterns, singular values, unit-norm PC scores
  3. SVD of cross-product matrix (Y_pcs.T @ X_pcs) → canonical correlations (mu),
     CCA rotations (r for Y, s for X)
  4. Predict: project new X → normalize by svx → rotate by s → scale by mu →
     rotate by r → scale by svy → project through eofy → add mean
"""
import numpy as np
import xarray as xr
from scipy.stats import kendalltau, norm
from .base import MethodBase
from ..registry import register_method


def _svd_pca(X, n_components):
    """SVD-based PCA matching CPT's get_pcs().

    Returns:
        eof: (n_features, n_components) — spatial loadings (V from SVD)
        ts: (n_components, n_samples) — unit-norm PC scores (U.T from SVD)
        sv: (n_components,) — singular values
    """
    # X is (n_samples, n_features), already centered+weighted
    # CPT does gesdd on (m, n) = (features, samples), i.e., X.T
    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
    # U: (features, min(f,s)) — EOF loadings
    # s: singular values
    # Vt: (min(f,s), samples) — unit-norm PC time series
    nc = min(n_components, len(s))
    return U[:, :nc], Vt[:nc, :], s[:nc]


_SV_RTOL = 1e-10


def _project_by_sv(num, sv):
    """Divide EOF projections by singular values, dropping degenerate modes.

    Modes whose singular value is negligible relative to the leading one carry no real variance;
    dividing by them amplifies rounding noise without bound. Observed in practice: a near-rank-
    deficient predictor produced CPT leverages of ~1e91 (statistical leverage is bounded in [0, 1],
    so these are numerical artefacts). Because ``seasonal_mme``'s pooled aggregation averages
    leverages across models, one such model inflated the MME predictive variance to effectively
    infinity, collapsing every tercile forecast to a constant [0.5, 0, 0.5] — i.e. GROC exactly
    0.500 regardless of the predictor. Zeroing the degenerate modes (the same policy as
    ``numpy.linalg.pinv``'s rcond cutoff) keeps the projection finite and well-posed.
    """
    sv = np.asarray(sv, dtype=float)
    num = np.asarray(num, dtype=float)
    out = np.zeros(num.shape, dtype=float)
    if sv.size == 0:
        return out
    smax = float(np.max(np.abs(sv))) if np.isfinite(sv).any() else 0.0
    if not np.isfinite(smax) or smax <= 0.0:
        return out
    keep = np.abs(sv) > smax * _SV_RTOL
    out[keep] = num[keep] / sv[keep]
    return out


@register_method("cca")
class CCAMethod(MethodBase):
    def __init__(self, n_modes=3, x_eof_modes=None, y_eof_modes=None,
                 cca_modes=None, standardize=False,
                 transform_predictand=None, tailoring=None,
                 drymask_threshold=None, synchronous_predictors=True):
        self.n_modes = n_modes
        self.x_eof_modes = x_eof_modes
        self.y_eof_modes = y_eof_modes
        self.cca_modes = cca_modes
        self.standardize = standardize
        # --- CPT_ARGS parity (§7) ---
        # transform_predictand: None | "Empirical" (rank -> normal-score round
        #   trip on the predictand, inverted after predict). "Gamma" is deferred
        #   (Gamma on demeaned / zero-inflated precip is ill-posed; see §5).
        # tailoring: None | "Anomaly". "Anomaly" returns the forecast as an
        #   anomaly from climatology (the long-term mean is not added back).
        # drymask_threshold: float | None. Predictand cells whose climatological
        #   mean is below this are excluded from the SVD (masked *before* the fit).
        # synchronous_predictors: assumed True (predictor and predictand share a
        #   year axis). Exposed for CPT-config parity; no behaviour change.
        self.transform_predictand = transform_predictand
        self.tailoring = tailoring
        self.drymask_threshold = drymask_threshold
        self.synchronous_predictors = synchronous_predictors

    def fit(self, hindcast, obs, **kwargs):
        x_eof_modes = kwargs.get("x_eof_modes", self.x_eof_modes)
        y_eof_modes = kwargs.get("y_eof_modes", self.y_eof_modes)
        cca_modes = kwargs.get("cca_modes", self.cca_modes) or kwargs.get("n_modes", self.n_modes)
        transform_predictand = kwargs.get("transform_predictand", self.transform_predictand)
        drymask_threshold = kwargs.get("drymask_threshold", self.drymask_threshold)
        # Stash for predict() (CV always calls fit before predict).
        self.transform_predictand_ = transform_predictand
        self.tailoring_ = kwargs.get("tailoring", self.tailoring)
        if transform_predictand == "Gamma":
            raise NotImplementedError(
                "transform_predictand='Gamma' is deferred (Gamma on demeaned / "
                "zero-inflated precip is ill-posed; see §5). Use 'Empirical' or None."
            )
        if transform_predictand not in (None, "Empirical"):
            raise ValueError(
                f"transform_predictand must be None or 'Empirical' (V1); got "
                f"{transform_predictand!r}."
            )

        gcm_mean = hindcast.mean("member")
        n_years = len(gcm_mean.year)

        X = gcm_mean.values.reshape(n_years, -1)
        Y = obs.values.reshape(n_years, -1)

        # Mask out columns that are all-NaN.
        self.x_valid_ = ~np.isnan(X).all(axis=0)
        self.y_valid_ = ~np.isnan(Y).all(axis=0)
        # Drymask (§7): exclude predictand cells whose climatological mean is
        # below the threshold *before* the fit, so they never enter the SVD.
        if drymask_threshold is not None:
            with np.errstate(invalid="ignore"):
                y_clim = np.nanmean(Y, axis=0)
            self.y_valid_ = self.y_valid_ & (y_clim >= drymask_threshold)
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        X = np.nan_to_num(X, nan=np.nanmean(X))
        Y = np.nan_to_num(Y, nan=np.nanmean(Y))

        # transform_predictand="Empirical": map each predictand column to normal
        # scores (rank -> plotting position -> Gaussian quantile) before fitting;
        # store the sorted values per column to invert after predict.
        if transform_predictand == "Empirical":
            self._y_sorted = np.sort(Y, axis=0)
            self._y_plot_pos = (np.arange(1, n_years + 1) - 0.5) / n_years
            ranks = Y.argsort(axis=0).argsort(axis=0) + 1
            Y = norm.ppf((ranks - 0.5) / n_years)

        # Center
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)
        X_c = X - self.x_mean_
        Y_c = Y - self.y_mean_

        # Standardize: divide by per-column sample std (ddof=1).
        # CPT does this before lat-weighting and SVD. Confirmed by matching
        # EOF explained variances to 6 decimal places vs CPT Fortran output.
        if self.standardize:
            self.x_std_ = X_c.std(axis=0, ddof=1)
            self.x_std_[self.x_std_ < 1e-20] = 1.0
            self.y_std_ = Y_c.std(axis=0, ddof=1)
            self.y_std_[self.y_std_ < 1e-20] = 1.0
            X_c = X_c / self.x_std_
            Y_c = Y_c / self.y_std_
        else:
            self.x_std_ = None
            self.y_std_ = None

        # Latitude area weighting (CPT latitude_weight before SVD)
        x_lats = np.repeat(gcm_mean.lat.values, len(gcm_mean.lon))
        y_lats = np.repeat(obs.lat.values, len(obs.lon))
        self.x_wt_ = np.sqrt(np.cos(np.deg2rad(x_lats)))[self.x_valid_]
        self.y_wt_ = np.sqrt(np.cos(np.deg2rad(y_lats)))[self.y_valid_]
        X_c = X_c * self.x_wt_
        Y_c = Y_c * self.y_wt_

        # EOF truncation via SVD (matching CPT get_pcs)
        if x_eof_modes is None:
            x_eof_modes = min(n_years - 1, X.shape[1], 10)
        if y_eof_modes is None:
            y_eof_modes = min(n_years - 1, Y.shape[1], 10)

        self.eofx_, self.tsx_, self.svx_ = _svd_pca(X_c, x_eof_modes)
        self.eofy_, self.tsy_, self.svy_ = _svd_pca(Y_c, y_eof_modes)
        # A rank-0 predictor (no interannual variance at all — e.g. a field that arrived
        # zero-filled from a failed OPeNDAP transfer) makes every projection 0/0. Silently
        # returning NaN forecasts hides a data problem as a modelling result, so fail loudly.
        if self.svx_.size == 0 or not np.isfinite(self.svx_).any() \
                or float(np.max(np.abs(self.svx_))) <= 0.0:
            raise ValueError(
                "CCA fit: the predictor has no interannual variance (all X singular values are "
                "zero). This usually means the predictor field was fetched as all-zeros or "
                "constant — check the source data rather than the model configuration."
            )
        nxe = len(self.svx_)
        nye = len(self.svy_)

        # CCA via SVD of cross-product matrix (CPT cca.F95 L111-112)
        # tsx and tsy are unit-norm: shape (n_eof, n_years)
        ce = self.tsy_ @ self.tsx_.T  # (nye, nxe)
        U_ce, mu, Vt_ce = np.linalg.svd(ce, full_matrices=False)

        ncc = min(cca_modes, nxe, nye, n_years - 1)
        self.ncc_ = ncc
        self.mu_ = mu[:ncc]             # canonical correlations
        self.r_ = U_ce[:, :ncc]         # Y CCA weights (nye, ncc)
        self.s_ = Vt_ce[:ncc, :]        # X CCA weights transposed (ncc, nxe)

        self.n_train_ = n_years
        self.x_eof_modes_ = nxe
        self.y_eof_modes_ = nye
        self.predictor_shape_ = gcm_mean.isel(year=0).shape
        self.predictor_coords_ = {"lat": gcm_mean.lat, "lon": gcm_mean.lon}
        self.predictand_shape_ = obs.isel(year=0).shape
        self.predictand_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def leverage(self, forecast):
        """CPT-compatible leverage (cca.F95 L602, L618-620).

        xvp = 1/n + Sum(prjc(1:ncc))**2
        where prjc = s @ (eofx.T @ x_anom / svx)
        """
        n = self.n_train_
        x = forecast.mean("member").values.reshape(1, -1)
        x = x[:, self.x_valid_]
        x = np.nan_to_num(x, nan=np.nanmean(x))
        x_anom = x - self.x_mean_
        if self.x_std_ is not None:
            x_anom = x_anom / self.x_std_
        x_anom = x_anom * self.x_wt_
        rwk = _project_by_sv(self.eofx_.T @ x_anom.ravel(), self.svx_)
        prjc = self.s_ @ rwk
        return 1.0 / n + float(np.sum(prjc)) ** 2

    def predict(self, forecast, **kwargs):
        """CPT-compatible prediction (cca.F95 L605-638).

        For each member:
          1. x_anom = (x - xm) [/ x_std] * lat_wt
          2. rwk = eofx.T @ x_anom          (project onto X EOFs)
          3. rwk = rwk / svx                 (normalize to unit variance)
          4. prjc = s @ rwk                  (project onto CCA modes)
          5. prjc = prjc * mu               (scale by canonical correlations)
          6. rwk = r @ prjc                  (back-project to Y EOF space)
          7. rwk = rwk * svy                 (scale by Y singular values)
          8. fcast = eofy @ rwk              (back-project to Y grid space)
          9. fcast = fcast / lat_wt [* y_std] + ym  (undo transforms, add mean)
        """
        forecast_grid = (forecast.sizes["lat"], forecast.sizes["lon"])
        if forecast_grid != self.predictor_shape_:
            raise ValueError(
                f"forecast grid shape (lat={forecast_grid[0]}, lon={forecast_grid[1]}) "
                f"does not match training predictor grid shape "
                f"(lat={self.predictor_shape_[0]}, lon={self.predictor_shape_[1]})"
            )
        results = []
        for m in range(len(forecast.member)):
            x = forecast.isel(member=m).values.reshape(1, -1)
            x = x[:, self.x_valid_]
            x = np.nan_to_num(x, nan=np.nanmean(x))

            x_anom = x.ravel() - self.x_mean_
            if self.x_std_ is not None:
                x_anom = x_anom / self.x_std_
            x_anom = x_anom * self.x_wt_

            rwk = _project_by_sv(self.eofx_.T @ x_anom, self.svx_)
            prjc = self.s_ @ rwk * self.mu_
            rwk_y = self.r_ @ prjc * self.svy_
            fcast = self.eofy_ @ rwk_y

            y_pred_valid = fcast / self.y_wt_
            if self.y_std_ is not None:
                y_pred_valid = y_pred_valid * self.y_std_
            # tailoring="Anomaly": leave the forecast as an anomaly (don't add
            # the climatological mean back). Default adds it back (full field).
            if getattr(self, "tailoring_", None) != "Anomaly":
                y_pred_valid = y_pred_valid + self.y_mean_

            # Invert the empirical predictand transform: normal score -> uniform
            # (Gaussian CDF) -> data value (interpolated on the stored per-cell
            # empirical distribution). Bounded to the observed range.
            if getattr(self, "transform_predictand_", None) == "Empirical":
                if getattr(self, "tailoring_", None) == "Anomaly":
                    raise NotImplementedError(
                        "transform_predictand='Empirical' with tailoring='Anomaly' "
                        "is unsupported (the inverse transform needs the full field)."
                    )
                u = norm.cdf(y_pred_valid)
                inv = np.empty_like(y_pred_valid)
                for j in range(y_pred_valid.shape[0]):
                    inv[j] = np.interp(u[j], self._y_plot_pos, self._y_sorted[:, j])
                y_pred_valid = inv

            y_full = np.full(len(self.y_valid_), np.nan)
            y_full[self.y_valid_] = y_pred_valid
            results.append(y_full.reshape(self.predictand_shape_))

        return xr.DataArray(
            np.stack(results),
            dims=["member", "lat", "lon"],
            coords={"member": forecast.member, **self.predictand_coords_},
        )


def _capped_mode_ranges(x_eof_range, y_eof_range, cca_range, n_years, window):
    """Cap the upper search bound by what the sample supports.

    An EOF/CCA fit needs strictly fewer modes than training samples, and the
    downstream CPT Student-t dof is ``n - modes - 1``. With ``window`` years held
    out per CV fold, cap each range at ``n_years - window - 2`` (never below 1)
    so auto-selection on a short hindcast can't pick a high-mode combo that
    overfits or drives the dof to <= 1 (which silently drops the model). On a
    long hindcast the cap is generous and the (1, 10) default is unaffected.
    """
    max_modes = max(1, n_years - max(window, 1) - 2)

    def _cap(r):
        lo, hi = r
        return (min(lo, max_modes), min(hi, max_modes))

    return _cap(x_eof_range), _cap(y_eof_range), _cap(cca_range)


def _north_modes(field, ceiling, min_frac_of_mean=1.0):
    """Retained EOF count for a (year, lat, lon[, member]) field by eigenvalue test.

    Reproduces ACMAD's ``compute_scree_north_test.gs`` (the CPT.x dry-run
    mode-selection step): do an area-weighted EOF of the field, then keep the
    leading modes that are both (a) above the *scree* line — individual explained
    variance exceeds the average ``min_frac_of_mean / n`` — and (b) *North-
    separated* from the next mode, i.e. the eigenvalue gap exceeds the sampling
    error ``dlambda = lambda * sqrt(2/N)`` (North et al. 1982, MWR 110, 699-706).
    Capped at ``ceiling`` and at least 1.

    Anomaly centering and ``sqrt(cos(lat))`` weighting follow the exact
    conventions of ``_svd_pca`` / ``CCAMethod.fit``, so the eigenvalues examined
    here are the ones the CCA fit would produce. A ``member`` dim is reduced by
    its mean first (matching what the CCA is fit on). Cost: one SVD.
    """
    da = field.mean("member") if "member" in field.dims else field
    da = da.transpose("year", "lat", "lon")
    n = da.sizes["year"]
    w = np.sqrt(np.cos(np.deg2rad(np.repeat(da.lat.values, da.sizes["lon"]))))
    a = da.values.reshape(n, -1)
    good = np.all(np.isfinite(a), axis=0)
    a = (a[:, good] - a[:, good].mean(0)) * w[good]
    lam = np.linalg.svd(a, compute_uv=False) ** 2
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1
    frac = lam / lam.sum()
    dl = lam * np.sqrt(2.0 / n)                     # North sampling error
    keep = 1
    for i in range(min(len(lam) - 1, ceiling)):
        if frac[i] < min_frac_of_mean / len(lam):   # scree: below-average variance
            break
        keep = i + 1
        if (lam[i] - lam[i + 1]) <= dl[i]:          # North: not separated from next
            break
    return max(1, min(keep, ceiling))


def select_modes_north(gcm, obs, x_ceiling=10, y_ceiling=10, min_frac_of_mean=1.0):
    """Eigenvalue-spectrum (Scree + North's rule) mode selection.

    Selects the EOF truncations *before any skill measurement*, the way CPT.x's
    dry-run does: X modes from the (ensemble-mean) predictor hindcast, Y modes
    from the predictand, CCA modes = min(X, Y). Yields the parsimonious 2-5
    modes per field that match ACMAD's operational selection, at the cost of one
    SVD per field (milliseconds).

    Contrast with :func:`select_modes` (cross-validated Kendall's-tau grid
    search): the tau search is CPT-compatible for skill-rich regimes, but on
    weak-signal seasonal fields the tau signal is noise and the search
    over-selects (measured: picked 10/10/10 at tau = -0.057, reproducing the
    over-fit it was meant to prevent, at ~27 min/member vs <1 s here). Use
    ``"north"`` for weak-signal seasonal fields, ``"cpt"``/``"auto"`` when CV
    skill is real.

    Returns ``(x_eof, y_eof, cca)`` fixed mode counts.
    """
    x_eof = _north_modes(gcm, x_ceiling, min_frac_of_mean)
    y_eof = _north_modes(obs, y_ceiling, min_frac_of_mean)
    return x_eof, y_eof, min(x_eof, y_eof)


def select_modes(gcm, obs, years, window, x_eof_range=(1, 10), y_eof_range=(1, 10),
                 cca_range=(1, 10), fallback_modes=None):
    """CPT-compatible mode auto-selection via cross-validated Kendall's tau.

    Matches CPT's cv_cca (cca.F95 L223-539) + goodness (scores.F95 L3589-3726):
    1. For each CV fold, fit CCA for every mode combination, predict held-out year
    2. Compute average Kendall's tau across grid points for each combination
    3. Return the combination with highest goodness

    The upper end of each range is capped by the sample size (see
    ``_capped_mode_ranges``) so a short hindcast can't select a high-mode combo
    that overfits or drives the downstream Student-t dof to <= 1. ``fallback_modes``
    (x_eof, y_eof, cca) is returned when no combination yields a finite goodness
    instead of raising, if supplied.

    Returns (best_x_eof, best_y_eof, best_cca, goodness_value, cv_predictions, leverages)
    """
    from ..cv import loyo

    # Cap the search to what the sample size supports (short-hindcast guard).
    capped = _capped_mode_ranges(x_eof_range, y_eof_range, cca_range, len(years), window)
    if capped != (tuple(x_eof_range), tuple(y_eof_range), tuple(cca_range)):
        print(f"  Mode selection: capping search ranges to sample size "
              f"({len(years)} yrs, window={window}): "
              f"x_eof={capped[0]}, y_eof={capped[1]}, cca={capped[2]}")
    x_eof_range, y_eof_range, cca_range = capped

    # Enumerate all valid mode combinations
    combos = []
    for xe in range(x_eof_range[0], x_eof_range[1] + 1):
        for ye in range(y_eof_range[0], y_eof_range[1] + 1):
            for cc in range(cca_range[0], min(cca_range[1], xe, ye) + 1):
                combos.append((xe, ye, cc))

    n_combos = len(combos)
    n_years = len(years)

    # Collect CV predictions for every combo: shape (n_combos, n_years, n_grid)
    obs_flat = obs.values.reshape(n_years, -1)
    valid_mask = ~np.isnan(obs_flat).all(axis=0)
    preds_all = np.full((n_combos, n_years, valid_mask.sum()), np.nan)
    levs_all = np.full((n_combos, n_years), np.nan)

    print(f"  Mode selection: {n_combos} combinations x {n_years} CV folds...")
    for fold_idx, (train_yrs, test_yr) in enumerate(loyo(years, window=window)):
        yr_idx = years.index(test_yr)
        forecast = gcm.sel(year=[test_yr]).isel(year=0, drop=True)

        for ci, (xe, ye, cc) in enumerate(combos):
            try:
                m = CCAMethod(x_eof_modes=xe, y_eof_modes=ye, cca_modes=cc)
                m.fit(gcm.sel(year=train_yrs), obs.sel(year=train_yrs))
                pred = m.predict(forecast).mean("member")
                pred_flat = pred.values.reshape(-1)
                preds_all[ci, yr_idx, :] = pred_flat[valid_mask]
                levs_all[ci, yr_idx] = m.leverage(forecast)
            except Exception:
                # CCA can fail for degenerate mode counts; leave as NaN
                pass

    # Compute goodness for each combo: average Kendall's tau across grid points
    # (CPT igood=3, scores.F95 L3675-3679)
    obs_valid = obs_flat[:, valid_mask]
    n_grid = obs_valid.shape[1]
    best_goodness = -np.inf
    best_idx = 0

    for ci in range(n_combos):
        preds = preds_all[ci]
        if np.isnan(preds).all():
            continue
        tau_sum = 0.0
        n_valid_pts = 0
        for gi in range(n_grid):
            o = obs_valid[:, gi]
            p = preds[:, gi]
            mask = np.isfinite(o) & np.isfinite(p)
            if mask.sum() < 4:
                continue
            tau, _ = kendalltau(p[mask], o[mask])
            if np.isfinite(tau):
                tau_sum += tau
                n_valid_pts += 1
        if n_valid_pts > 0:
            g = tau_sum / n_valid_pts
            if g > best_goodness:
                best_goodness = g
                best_idx = ci

    if not np.isfinite(best_goodness):
        if fallback_modes is None:
            raise ValueError(
                "CCA mode selection found no finite Kendall goodness value; "
                "provide fallback_modes=(x_eof, y_eof, cca) to continue."
            )
        fallback_modes = tuple(fallback_modes)
        if fallback_modes not in combos:
            raise ValueError(
                f"fallback_modes={fallback_modes} is outside searched mode "
                f"combinations."
            )
        best_idx = combos.index(fallback_modes)
        best_goodness = np.nan

    xe, ye, cc = combos[best_idx]
    print(f"  Optimal modes: x_eof={xe}, y_eof={ye}, cca={cc} (Kendall tau={best_goodness:+.4f})")

    # Reconstruct xarray predictions and leverages for the best combo
    best_preds_flat = preds_all[best_idx]
    best_preds = np.full((n_years, *obs.isel(year=0).shape), np.nan)
    best_preds.reshape(n_years, -1)[:, valid_mask] = best_preds_flat

    cv = xr.DataArray(best_preds, dims=["year", "lat", "lon"],
                      coords={"year": years, "lat": obs.lat, "lon": obs.lon})
    leverages = levs_all[best_idx]

    return xe, ye, cc, best_goodness, cv, leverages
