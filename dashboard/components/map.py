"""Interactive Bay of Bengal map component for OceanEmbed dashboard."""

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

DOMAIN_LAT_MIN = 5.0
DOMAIN_LAT_MAX = 23.0
DOMAIN_LON_MIN = 80.0
DOMAIN_LON_MAX = 100.0


def validate_coordinates(lat: float, lon: float) -> tuple[bool, str]:
    """Validate whether coordinates fall inside the supported Bay of Bengal demo domain."""
    if not (DOMAIN_LAT_MIN <= lat <= DOMAIN_LAT_MAX):
        return (
            False,
            f"Latitude {lat:.4f}°N is outside the Bay of Bengal demo domain ({DOMAIN_LAT_MIN:.1f}°N – {DOMAIN_LAT_MAX:.1f}°N).",
        )
    if not (DOMAIN_LON_MIN <= lon <= DOMAIN_LON_MAX):
        return (
            False,
            f"Longitude {lon:.4f}°E is outside the Bay of Bengal demo domain ({DOMAIN_LON_MIN:.1f}°E – {DOMAIN_LON_MAX:.1f}°E).",
        )
    return (True, "")


def render_bay_of_bengal_map(
    selected_lat: float,
    selected_lon: float,
    key: str = "bay_of_bengal_map",
) -> tuple[float | None, float | None]:
    """Render interactive Folium map centered on the Bay of Bengal.

    Returns:
        (clicked_lat, clicked_lon) if user clicked inside domain, else (None, None).
    """
    m = folium.Map(
        location=[14.0, 88.0],
        zoom_start=5,
        min_zoom=4,
        max_zoom=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Add Domain Bounding Box for Bay of Bengal
    folium.Rectangle(
        bounds=[[DOMAIN_LAT_MIN, DOMAIN_LON_MIN], [DOMAIN_LAT_MAX, DOMAIN_LON_MAX]],
        color="#1f77b4",
        weight=2,
        fill=True,
        fill_color="#2980b9",
        fill_opacity=0.08,
        dash_array="6, 6",
        tooltip="Supported Bay of Bengal Domain (5°N–23°N, 80°E–100°E)",
    ).add_to(m)

    # Place marker at currently selected location if valid
    is_valid, _ = validate_coordinates(selected_lat, selected_lon)
    if is_valid:
        folium.Marker(
            location=[selected_lat, selected_lon],
            tooltip=f"Selected Point: {selected_lat:.4f}°N, {selected_lon:.4f}°E",
            popup=folium.Popup(
                f"<b>OceanEmbed Target</b><br>Lat: {selected_lat:.4f}°N<br>Lon: {selected_lon:.4f}°E",
                max_width=200,
            ),
            icon=folium.Icon(color="darkblue", icon="tint", prefix="fa"),
        ).add_to(m)

    # Render in Streamlit
    map_data = st_folium(
        m,
        width="100%",
        height=380,
        key=key,
        returned_objects=["last_clicked"],
    )

    if map_data and map_data.get("last_clicked"):
        c_lat = round(map_data["last_clicked"]["lat"], 4)
        c_lon = round(map_data["last_clicked"]["lng"], 4)
        return c_lat, c_lon

    return None, None
