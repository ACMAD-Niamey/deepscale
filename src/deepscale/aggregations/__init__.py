"""Daily-rainfall aggregations: season timing and dry spells.

Access is module-qualified, following ``deepscale.time`` and
``deepscale.tercile``::

    from deepscale import aggregations as agg

    onset = agg.onset(daily, season="MAM")
"""
from .spells import DrySpellResult, dry_spell
from .timing import ONSET_DEFAULTS, TimingResult, cessation, onset, season_length

__all__ = [
    "onset",
    "cessation",
    "season_length",
    "dry_spell",
    "TimingResult",
    "DrySpellResult",
    "ONSET_DEFAULTS",
]
