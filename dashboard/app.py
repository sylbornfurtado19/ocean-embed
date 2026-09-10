"""Streamlit dashboard for OceanEmbed.

TODO: implement a minimal UI for selecting lat/lon/date and plotting the predicted depth profile.
"""

from __future__ import annotations

import streamlit as st


def plot_depth_profile(lat: float, lon: float, date: str) -> None:
    """Plot the model-predicted temperature profile at the selected location and time."""
    # TODO: query model or cached results and render chart.
    raise NotImplementedError("TODO: implement dashboard profile plot.")


def main() -> None:
    """Render the OceanEmbed dashboard interface."""
    st.title("OceanEmbed Dashboard")
    st.write("Select a location and date to inspect the reconstructed depth profile.")

    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0)
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=85.0)
    date = st.date_input("Date")

    # TODO: connect controls to model predictions and plotting.
    st.caption("Dashboard logic is not implemented yet.")

    plot_depth_profile(float(lat), float(lon), str(date))


if __name__ == "__main__":
    main()
