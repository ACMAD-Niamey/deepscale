from .downscale import downscale
from .optimize import optimize
from .ensemble import ensemble, pool_ensembles
from .skill import skill, SkillReport
from .compare import skill_compare, ComparisonReport
from .pev import prediction_error_variance
from .flex_forecast import flex_forecast, FlexForecastResult
from .pipelines import seasonal_mme, SeasonalMMEResult
from .indices import Index
from .calibrate import calibrate, CalibrateResult, LogitConfig
from .climate import (
    accumulate,
    frequency_below,
    percent_of_normal,
    percentile_of,
    rank_of_record,
    seasonal_reduce,
    seasonal_stack,
)
from .analog import (
    AnalogSet, analogs_from_years, analogs_from_index, analogs_from_field, analogs_where,
)
from .completion import complete, CompletionResult
from .series import quantile_map, error_bounds, ErrorBounds
from .methods.smoothed_regression import seasonal_coefficients
from .io import write_terciles, tercile_mae
from .terrain import load_terrain
from .combine import combine_terciles, mask_by_skill, dry_mask
from .plotting.forecasts import plot_tercile_forecast as plot_terciles
from .plotting.scenarios import plot_accumulation_scenarios, plot_index_scatter
from .plotting.maps import natural_earth_borders, plot_field_map, plot_choropleth
from .plotting.style import TercileStyle
from .plotting.forecasts import plot_field, plot_tercile_comparison, render_styled_terciles
from . import methods     # trigger registration
from . import metrics     # trigger registration
from .metrics import loo_predict, loo_corr, permutation_test, fdr
from . import strategies  # trigger registration
from . import time        # calendar / season-step utilities
from . import aggregations  # daily-rainfall season timing and dry spells
from . import training     # train / inference separation (§10.2)
from .training import train

__all__ = [
    "downscale",
    "train",
    "optimize",
    "ensemble",
    "pool_ensembles",
    "skill",
    "SkillReport",
    "skill_compare",
    "ComparisonReport",
    "prediction_error_variance",
    "flex_forecast",
    "FlexForecastResult",
    "seasonal_mme",
    "SeasonalMMEResult",
    "Index",
    "calibrate",
    "CalibrateResult",
    "LogitConfig",
    "accumulate",
    "frequency_below",
    "percent_of_normal",
    "percentile_of",
    "rank_of_record",
    "seasonal_reduce",
    "seasonal_stack",
    "AnalogSet",
    "analogs_from_years",
    "analogs_from_index",
    "analogs_from_field",
    "analogs_where",
    "complete",
    "CompletionResult",
    "quantile_map",
    "error_bounds",
    "ErrorBounds",
    "seasonal_coefficients",
    "write_terciles",
    "tercile_mae",
    "load_terrain",
    "combine_terciles",
    "mask_by_skill",
    "dry_mask",
    "loo_predict",
    "loo_corr",
    "permutation_test",
    "fdr",
    "plot_terciles",
    "plot_accumulation_scenarios",
    "plot_index_scatter",
    "plot_field_map",
    "plot_choropleth",
    "natural_earth_borders",
    "TercileStyle",
    "plot_field",
    "render_styled_terciles",
    "plot_tercile_comparison",
]
