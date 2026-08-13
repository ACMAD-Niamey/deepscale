# Daily aggregations (`deepscale.aggregations`)

Rainy-season timing (onset, cessation, season length) and dry-spell statistics
from continuous daily rainfall. Everything here answers one family of question:
where, within a season, do specific rainfall patterns occur, read off a daily
time axis rather than the `(year, step, …)` season-aligned axis the rest of
deepscale works on.

Reference this module-qualified, following `deepscale.time` and
`deepscale.tercile`. Its functions are not in the top-level `__all__`:

```python
from deepscale import aggregations as agg

onset = agg.onset(daily, season="MAM")
```

Runnable end-to-end (needs `rosetta` installed, for the CHIRPS fetch): [../../../examples/onset_gha_mam.py](../../../examples/onset_gha_mam.py).

---

## The onset criterion

Onset is the first day satisfying a trigger and a guard. The defaults below are
a widely used definition for East African MAM seasons, and are what the
reference implementation this module was ported from uses. Nothing in the
module assumes them: every value is a plain keyword argument, so a different
regional definition is one call away. The five values are exported as
`agg.ONSET_DEFAULTS` for callers that want to read or diff them.

| Component | Rule | Default |
|---|---|---|
| Trigger | `accum_days` consecutive days total at least `accum_mm` | `accum_mm=20.0`, `accum_days=3` |
| Guard | no run of `dry_spell_days` or more days at or below `dry_spell_mm` falls entirely within the following `guard_days` | `dry_spell_mm=1.0`, `dry_spell_days=7`, `guard_days=21` |

The guard rejects false starts: a trigger that clears the accumulation but is
followed by a dry spell within the guard window does not count as onset.
`onset` searches for the first step where both hold at once.

---

## Signatures

```python
onset(daily, season, *, search_start=None, search_end=None,
      accum_mm=20.0, accum_days=3, dry_spell_mm=1.0, dry_spell_days=7,
      guard_days=21, time_dim="time") -> TimingResult
```

`daily` is continuous daily rainfall in mm/day with a datetime `time_dim`, not
pre-stacked: `season` is what carves it into years. Dims other than time
survive, so `(time, lat, lon)` observations and `(member, time, lat, lon)`
forecasts take the same path. `season` is anything `deepscale.time.season_bounds`
accepts. `search_start`/`search_end` are 0-based step bounds inside the season
restricting where a trigger may fire; they do not restrict the data available
to the guard, which always reads into the tail. `None` means the season
boundary.

```python
cessation(daily, season, *, after, dry_spell_mm=1.0, dry_spell_days=7,
          time_dim="time") -> TimingResult
```

Cessation is the first qualifying dry spell after `after`, which is required
and has no default: pass the `onset` result for a per-cell floor, or an
integer step for a uniform one. A dry spell before onset is not a cessation,
and the onset criterion explicitly tolerates pre-onset dry spells, so
searching from an arbitrary fixed point routinely returns the wrong answer.
Where `after` is a `TimingResult` with no onset, the comparison against NaN is
False everywhere and cessation is correctly absent too.

```python
season_length(onset, cessation) -> xr.DataArray
```

Days from onset to cessation (`cessation.step - onset.step`), NaN wherever
either end is missing. A season that started but had not ceased within the
available window has a length of NaN rather than a truncated value: the answer
is unknown, not short.

```python
dry_spell(daily, season, *, dry_spell_mm=1.0, dry_spell_days=7,
          time_dim="time") -> DrySpellResult
```

Longest dry run and count of qualifying dry runs, per season. A run is dry
when each of its days receives `dry_spell_mm` or less. The defaults are the
dry-spell half of the default onset criterion. No tail is stacked: unlike
onset and cessation, this question is entirely contained within the season.

---

## `TimingResult` and `DrySpellResult`

```python
@dataclass
class TimingResult:
    step: xr.DataArray       # 0-based days since season start
    date: xr.DataArray       # calendar date, NaT where the event did not happen
    occurred: xr.DataArray   # 1.0 / 0.0 / NaN, see below
    params: dict
```

`step` is on the same axis as `deepscale.time.season_step`: leap-safe, and
correct for seasons crossing the new year, neither of which day-of-year is.
Its dims are `(year, …)`, so it can be used as a predictand anywhere the rest
of deepscale takes one.

One caveat before converting it: `to_tercile_cv` reads NaN as missing data,
and a NaN `step` here can also mean the season failed, which is a fourth
outcome rather than a low tail of the distribution. Decide what a failed season
should become (dropped, or its own category) before the tercile step, rather
than letting it be silently pooled with ocean cells. `occurred` is the field
that tells the two apart: 0.0 is a failed season, NaN is no data.

`occurred` is a float rather than a bool because there are three states, and a
bool array cannot hold NaN, so the three states would collapse to two:

| State | `step` | `occurred` |
|---|---|---|
| Timing event found | step index | 1.0 |
| Season failed, no qualifying event | NaN | 0.0 |
| No data, for example ocean | NaN | NaN |

Only in-season days decide which of the last two applies. A year whose season
falls entirely outside the record reads as no data even when the guard tail
overlaps the record, because tail data is not evidence that the season was
observed. `dry_spell` drops such a year outright, so the two entry points
agree about which years were observed.

```python
@dataclass
class DrySpellResult:
    longest: xr.DataArray   # length of the longest dry run
    count: xr.DataArray     # number of runs reaching dry_spell_days
    params: dict
```

`longest` is the length of the longest run at or below the threshold, whatever
its length. `count` is the number of distinct runs reaching `dry_spell_days`.
Both are NaN for cells with no data, and 0.0 for cells that simply had no dry
days: the same absent-data-versus-nothing-happened distinction `occurred`
draws for timing.

---

## Record coverage requirements

Input must be daily. All three functions call `deepscale.time.infer_cadence` on
the time coordinate and raise `ValueError` for anything else, rather than
warning: accumulation days, dry-spell length and the guard window are all
counted in days, so a dekadal product such as `obs/chirps-v2-dekadal-rhiza` has
no meaningful answer to return.

A rolling window question also needs data past the point it is asked about, or
the question cannot be answered. Each function needs a different amount of
`daily` past the season end, and each warns rather than silently returning a
wrong number when the record is too short:

| Function | Must extend past the season end by | What is unverifiable if it does not |
|---|---|---|
| `onset` | `accum_days + guard_days` | a late trigger's guard window |
| `cessation` | `dry_spell_days` | a dry spell beginning near the season end |
| `dry_spell` | nothing extra, but the season itself must be fully covered | any run spanning where the record cuts off |

Coverage is checked over every day each year needs, not only at the record's
last stamp, because a record can fail to cover a season in three ways:

| Shortfall | What the record does | What it reads as without the warning |
|---|---|---|
| ends early | stops before the season end plus that function's tail | a season that failed late |
| starts late | begins after the season has already started | an event on the first day of data |
| interior gaps | drops days inside the span | a season that failed, since a window spanning a gap fails closed |

This is a property of the time axis rather than of any one cell, so it is one
warning per call naming the affected year(s) and which shortfalls were found,
never one per grid cell. The ends-early case also names the date the record
would need to reach. This one is `onset`'s, with the default criterion
(`accum_days + guard_days = 24`):

```
daily data ends 2015-05-31 and does not cover the 24-day guard window for
season year(s) [2015]. Late-season triggers there cannot be verified and are
reported as no onset. Supply daily data through at least 2015-06-24.
```

`cessation`'s warning names the "dry-spell window" instead and reports the
unconfirmable cessations as absent rather than guessing. `dry_spell`'s warning
is different in kind: `seasonal_stack` NaN-pads a mid-season gap, and NaN
counts as not-dry, so a run spanning the cut is silently truncated rather than
producing NaN: `longest` and `count` under-report instead of reporting no
data. Identical rainfall at two record lengths gives two different answers,
which is exactly why this one warns even though the numbers stay finite.

---

## Cessation past the season's last index

`cessation` can legitimately return a step one day past the season's last
index, when the season rains through its own final day and the first
qualifying dry spell only starts in the tail. `season_length` can therefore
equal the full season width (92 for MAM, at onset step 0). This is correct
behaviour, not a leak: the coupling `tail_days == dry_spell_days` in
`cessation`'s call to `seasonal_stack` is what bounds a detectable run start
to at most one step past the season's last index. Widening the tail without
widening the dry-spell criterion to match would let cessation drift further
outside the season.

---

## Provenance attrs

Every returned array (`step`, `date`, `occurred` on `TimingResult`; `longest`,
`count` on `DrySpellResult`; the `season_length` array itself) carries the
resolved parameters as `.attrs`, not just on the dataclass, so provenance
survives `to_netcdf` into whatever file gets shipped. `season` and
`deepscale_version` are always included, alongside each function's criterion
parameters:

| Function | Recorded parameters |
|---|---|
| `onset` | `accum_mm`, `accum_days`, `dry_spell_mm`, `dry_spell_days`, `guard_days`, `search_start`, `search_end` |
| `cessation` | `dry_spell_mm`, `dry_spell_days` |
| `dry_spell` | `dry_spell_mm`, `dry_spell_days` |
| `season_length` | every key from its `onset` argument, plus `cessation_dry_spell_mm`, `cessation_dry_spell_days` |

`season_length` is produced by two criteria, so it records both. Cessation's
keys are prefixed `cessation_` because `dry_spell_mm` and `dry_spell_days` name
a different rule on each side, and a flat merge would let one overwrite the
other. `season` and `deepscale_version` describe the analysis rather than
either criterion, so they are recorded once, from the `onset` argument. The
full key set is:

| Key | From |
|---|---|
| `accum_mm`, `accum_days`, `dry_spell_mm`, `dry_spell_days`, `guard_days`, `search_start`, `search_end`, `onset_definition` | the `onset` argument, unprefixed |
| `cessation_dry_spell_mm`, `cessation_dry_spell_days` | the `cessation` argument, prefixed |
| `season`, `deepscale_version` | shared, recorded once |

Each of these is recorded whether or not the caller passed it explicitly: a
default is otherwise invisible at the call site, so it is written into the
output instead.

Two keyword arguments that the affected functions accept are deliberately
absent from `.attrs`, not omitted by oversight:

| Argument | Functions | Why it is not recorded |
|---|---|---|
| `time_dim` | `onset`, `cessation`, `dry_spell` | names which dimension of the input holds time, describing the input's layout rather than the criterion that produced the result |
| `after` | `cessation` | a per-cell field, not a scalar, so it has no meaningful attribute form; recover it from the onset result that produced it |

`onset`'s output additionally carries `onset_definition`, `"default"` if
every one of `accum_mm`, `accum_days`, `dry_spell_mm`, `dry_spell_days`, and
`guard_days` matches the defaults above, `"custom"` otherwise. A
result computed with the bare defaults and one computed by writing out those
same five values explicitly both read `"default"`: the label reflects the
resolved parameters, not how the call was written.

---

## Worked example

```python
from deepscale import aggregations as agg

onset = agg.onset(
    daily,
    season="MAM",
    search_start=None,
    search_end=None,
    accum_mm=20.0,
    accum_days=3,
    dry_spell_mm=1.0,
    dry_spell_days=7,
    guard_days=21,
    time_dim="time",
)
```

Every parameter above is already the default, so this call is exactly
equivalent to `agg.onset(daily, season="MAM")`.
