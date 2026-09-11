"""Profile data table and CSV export component for OceanEmbed dashboard."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def create_profile_dataframe(
    depths: list[float] | np.ndarray,
    temperature: list[float] | np.ndarray,
    sigma: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Construct a clean, structured DataFrame containing the 15-depth prediction outputs."""
    depths_arr = np.asarray(depths, dtype=np.float32)
    temp_arr = np.asarray(temperature, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)

    lower_90 = temp_arr - 1.645 * sigma_arr
    upper_90 = temp_arr + 1.645 * sigma_arr

    df = pd.DataFrame(
        {
            "depth_m": depths_arr,
            "temperature_c": np.round(temp_arr, 2),
            "sigma_c": np.round(sigma_arr, 2),
            "lower_90_c": np.round(lower_90, 2),
            "upper_90_c": np.round(upper_90, 2),
        }
    )
    return df


def render_profile_table(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    date_str: str,
) -> None:
    """Render the profile table with formatted display and download button."""
    display_df = df.copy()
    display_df.columns = [
        "Depth (m)",
        "Temperature (°C)",
        "Sigma (±°C)",
        "Lower 90% (°C)",
        "Upper 90% (°C)",
    ]

    st.dataframe(
        display_df.style.format(
            {
                "Depth (m)": "{:.0f}",
                "Temperature (°C)": "{:.2f}",
                "Sigma (±°C)": "{:.2f}",
                "Lower 90% (°C)": "{:.2f}",
                "Upper 90% (°C)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Prepare downloadable CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    filename = f"oceanembed_profile_{lat:.2f}N_{lon:.2f}E_{date_str}.csv"
    st.download_button(
        label="📥 Download Profile CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        help="Export depth-wise temperature and uncertainty estimates to CSV",
    )
