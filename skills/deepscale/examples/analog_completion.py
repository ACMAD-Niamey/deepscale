"""SMPG analog-year selection and scenario completion of a partial season.

The season-monitoring / seasonal-forecast-positioning workflow: part-way through
a rainy season, complete it with the past years that most resemble the state we
are in, and report where the likely end-of-season total falls in the record.

The pipeline is: season-align an archive -> pick analog years -> splice
observed-to-date (+ optional short-range forecast) + each analog's remainder
into one scenario per analog -> read the consensus total and its historical
percentile. Every step is dimension-agnostic, so the same code runs on a grid
`(time, lat, lon)`, on admin-unit series `(time, region)`, or on the single
station series shown here.

Inputs (any xarray source; rosetta produces these shapes):
- archive:      (time,) dekadal rainfall totals spanning many years (the record)
- observed:     (time,) this season's dekads observed so far
- next_30_days: (time,) a short-range forecast for the dekads after `observed`
- nino34:       (year,) a teleconnection index series (e.g. Index.reduce output)
"""

import deepscale as ds

SEASON = "JJAS"          # June-September; anything season_bounds accepts

# --- 1. Season-align the archive into (year, step, …) ---------------------
# `step` is the ordinal position within the season, so step 3 of 1997 and step 3
# of this year are comparable — which their calendar dates never are. This is the
# `climatology` shape `complete` requires.
clim = ds.seasonal_stack(archive, SEASON)            # (year, step)

# Season totals per year, for ranking the outcome at the end.
history = ds.accumulate(clim, dim="step")            # (year,) full-season totals

# --- 2. Choose analog years -----------------------------------------------
# (a) nearest neighbours to a forecast Niño3.4 of +1.4 (an El Niño-like state):
by_index = ds.analogs_from_index(nino34, target=1.4, n=12, metric="absolute")

# (b) a threshold predicate — every year that was at least weakly warm:
warm = ds.analogs_where(nino34 >= 0.5)

# Compose: warm AND close to the target, re-ranked on the mean of both scores,
# then keep the best 9. `&`/`|`/`.top(n)` work because every score is a distance.
analogs = (warm & by_index).top(9)
print(analogs)                                       # AnalogSet(9 years via and: [...])
print(analogs.candidates)                            # every year that was scored

# --- 3. Complete the partly-observed season -------------------------------
# Splices observed-to-date, then the forecast, then each analog's remainder on the
# season's ordinal step. One scenario per analog; the spread across them IS the
# uncertainty, and each member is a real season that happened.
result = ds.complete(
    observed, analogs,
    climatology=clim,
    season=SEASON,
    forecast=next_30_days,           # omit and rerun to isolate the forecast's contribution
    reduce="median",                 # consensus = median across analogs (float = quantile, or a callable)
    weights=analogs.weights("gaussian"),   # or "uniform" / "inverse_distance" / None
    percentile_reference=history,    # rank the consensus against full-season totals
)

# --- 4. Read the outcome --------------------------------------------------
print(result.consensus)              # the median end-of-season total
print(float(result.percentile))      # where it sits in the record, [0, 1]
print(result.totals)                 # (scenario,) one plausible total per analog year
print(result.segments.values)        # per-step "observed" / "forecast" / "analog"

# Positioning helpers work directly on the aligned archive too:
pct = ds.percentile_of(result.consensus, history, dim="year")     # same number as .percentile
rank = ds.rank_of_record(result.consensus, history, dim="year")   # 1 == driest on record
dry_freq = ds.frequency_below(clim.sel(year=analogs.years).sum("step"),
                              history, q=1/3)                       # below-normal frequency composite

# --- 5. Plot (needs the [plotting] extra) ---------------------------------
fig = ds.plot_accumulation_scenarios(result, climatology=clim,
                                     title=f"{SEASON} scenario completion")
fig.savefig("scenarios.png", dpi=200, bbox_inches="tight")

# Notes:
# - observed=None runs a pure analog projection (the season on history alone).
# - complete reduces along `step` only, so give it (time, lat, lon) for per-pixel
#   maps or (time, region) for per-district curves — no separate code path.
# - Calendar/season-step helpers are module-qualified: deepscale.time.season_step, etc.
