"""Rainy-season onset over the Greater Horn of Africa, from daily CHIRPS.

Applies the default MAM onset criterion across 2005-2023 and maps two things: the mean
onset date, and the fraction of years the season started at all.

The second map is the point of the demo. It is the product the reference
script cannot produce, because its onset_frac returns the same 0.0 for an ocean
cell as for a cell where every season failed. The three-state `occurred` field
separates them, so "how often does the season simply not arrive here" becomes
answerable.

Run: python examples/onset_gha_mam.py
"""
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rosetta

from deepscale import aggregations as agg
from deepscale.plotting.maps import plot_field_map

REGION = [-12.0, 23.0, 21.0, 52.0]   # [lat_s, lat_n, lon_w, lon_e], GHA
YEARS = (2005, 2023)
SEASON = "MAM"
OUT = "onset_gha_mam.png"


def step_to_label(step, year=2015):
    """Step 0 is 1 March. Render a mean step as a readable date."""
    start = dt.date(year, 3, 1)
    return (start + dt.timedelta(days=float(step))).strftime("%d %b")


def main():
    daily = rosetta.fetch(
        "obs/chirps-v3-daily-rhiza", "precip",
        region=REGION, hindcast=YEARS, months=[2, 3, 4, 5, 6],
    )["precip"]

    onset = agg.onset(daily, season=SEASON)
    cessation = agg.cessation(daily, season=SEASON, after=onset)
    length = agg.season_length(onset, cessation)
    spells = agg.dry_spell(daily, season=SEASON)

    mean_step = onset.step.mean("year")
    started = onset.occurred.mean("year")

    print(f"definition:       {onset.step.attrs['onset_definition']}")
    print(f"mean onset:       {step_to_label(float(mean_step.mean()))}")
    print(f"seasons started:  {float(started.mean()) * 100:.1f}% of cell-years")
    print(f"mean length:      {float(length.mean()):.1f} days")
    print(f"mean longest dry: {float(spells.longest.mean()):.1f} days")

    # plot_field_map uses a cartopy GeoAxes when cartopy is installed, and
    # expects a caller-supplied `ax=` to already carry that projection.
    # Mirror that here so the two calls below work whether or not cartopy is
    # present, rather than hard-depending on it.
    try:
        import cartopy.crs as ccrs
        subplot_kw = {"projection": ccrs.PlateCarree()}
    except ImportError:
        subplot_kw = None
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=subplot_kw)
    plot_field_map(
        mean_step, ax=axes[0], cmap="viridis_r",
        title=f"Mean {SEASON} onset, {YEARS[0]}-{YEARS[1]}",
        cbar_label="days after 1 March",
    )
    plot_field_map(
        started, ax=axes[1], cmap="RdYlBu", vmin=0.0, vmax=1.0,
        title="Fraction of years the season started",
        cbar_label="fraction",
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
