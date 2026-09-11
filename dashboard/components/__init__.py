"""OceanEmbed Dashboard Components package."""

from dashboard.components.map import render_bay_of_bengal_map, validate_coordinates
from dashboard.components.metrics import (
    render_embedding_panel,
    render_summary_metrics,
    render_surface_inputs,
)
from dashboard.components.plots import (
    render_climatology_decomposition_chart,
    render_temperature_profile_chart,
)
from dashboard.components.profile_table import create_profile_dataframe, render_profile_table
from dashboard.components.regime import render_regime_chart

__all__ = [
    "render_bay_of_bengal_map",
    "validate_coordinates",
    "render_temperature_profile_chart",
    "render_climatology_decomposition_chart",
    "render_regime_chart",
    "render_summary_metrics",
    "render_surface_inputs",
    "render_embedding_panel",
    "create_profile_dataframe",
    "render_profile_table",
]
