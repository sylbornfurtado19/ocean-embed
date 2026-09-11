"""Interactive Plotly plotting components for OceanEmbed dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def render_temperature_profile_chart(
    depths: list[float] | np.ndarray,
    temperature: list[float] | np.ndarray,
    sigma: list[float] | np.ndarray,
    show_uncertainty: bool = True,
) -> go.Figure:
    """Render interactive vertical temperature-depth profile with 90% predictive interval envelope."""
    depths_arr = np.asarray(depths, dtype=np.float32)
    temp_arr = np.asarray(temperature, dtype=np.float32)
    sigma_arr = np.asarray(sigma, dtype=np.float32)

    lower_90 = temp_arr - 1.645 * sigma_arr
    upper_90 = temp_arr + 1.645 * sigma_arr

    custom_data = np.stack([sigma_arr, lower_90, upper_90], axis=-1)

    fig = go.Figure()

    if show_uncertainty:
        # 1. Upper bound trace (invisible anchor line)
        fig.add_trace(
            go.Scatter(
                x=upper_90,
                y=depths_arr,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Upper 90% Bound",
            )
        )

        # 2. Lower bound trace with fill='tonextx' to create shaded confidence ribbon
        fig.add_trace(
            go.Scatter(
                x=lower_90,
                y=depths_arr,
                mode="lines",
                line=dict(width=0),
                fill="tonextx",
                fillcolor="rgba(2, 132, 199, 0.16)",
                hoverinfo="skip",
                name="90% Confidence Interval (±1.645σ)",
            )
        )

    # 3. Main temperature mean profile trace (Index 2 when uncertainty is active)
    fig.add_trace(
        go.Scatter(
            x=temp_arr,
            y=depths_arr,
            mode="lines+markers",
            line=dict(color="#0284c7", width=3.2),
            marker=dict(
                size=8,
                color="#0369a1",
                symbol="circle",
                line=dict(color="#ffffff", width=1.5),
            ),
            customdata=custom_data,
            hovertemplate=(
                "<span style='font-size:13px; font-weight:600; color:#0f172a;'>Depth: %{y:.0f} m</span><br>"
                "<b>Mean Temperature:</b> %{x:.2f} °C<br>"
                "<b>Predictive Uncertainty (σ):</b> ±%{customdata[0]:.2f} °C<br>"
                "<b>90% Confidence Bounds:</b> [%{customdata[1]:.2f} – %{customdata[2]:.2f}] °C"
                "<extra></extra>"
            ),
            name="OceanEmbed V2 Reconstructed Profile",
        )
    )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        title=dict(
            text="<b>Subsurface Ocean Temperature Profile (0 to 1000 m)</b>",
            font=dict(size=16, color="#0f172a"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text="<b>Temperature (°C)</b>", font=dict(size=12, color="#334155")),
            showgrid=True,
            gridcolor="#e2e8f0",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            title=dict(text="<b>Depth (m) — Increasing Downward</b>", font=dict(size=12, color="#334155")),
            autorange="reversed",  # Depth increases downwards
            showgrid=True,
            gridcolor="#e2e8f0",
            gridwidth=1,
            tickvals=[0, 50, 100, 150, 200, 300, 500, 700, 1000],
            tickfont=dict(size=11, color="#475569"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#334155"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=65, r=30, t=85, b=50),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=540,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(size=12, family="Inter, sans-serif", color="#0f172a"),
        ),
    )

    return fig


def render_climatology_decomposition_chart(
    depths: list[float] | np.ndarray,
    temperature: list[float] | np.ndarray,
    climatology: list[float] | np.ndarray,
    anomaly: list[float] | np.ndarray,
) -> go.Figure:
    """Render Plotly chart showing Climatology Prior + Predicted Anomaly = Final Reconstructed Profile."""
    depths_arr = np.asarray(depths, dtype=np.float32)
    temp_arr = np.asarray(temperature, dtype=np.float32)
    clim_arr = np.asarray(climatology, dtype=np.float32)
    anom_arr = np.asarray(anomaly, dtype=np.float32)

    fig = go.Figure()

    # 1. Climatology Prior (Background)
    fig.add_trace(
        go.Scatter(
            x=clim_arr,
            y=depths_arr,
            mode="lines+markers",
            line=dict(color="#64748b", width=2, dash="dash"),
            marker=dict(size=6, color="#475569", symbol="square"),
            hovertemplate="<b>Depth:</b> %{y:.0f} m<br><b>Climatology Prior:</b> %{x:.2f} °C<extra></extra>",
            name="Climatology Prior T_clim(z)",
        )
    )

    # 2. Predicted Temperature Anomaly
    fig.add_trace(
        go.Scatter(
            x=anom_arr,
            y=depths_arr,
            mode="lines+markers",
            line=dict(color="#f97316", width=2, dash="dot"),
            marker=dict(size=6, color="#ea580c", symbol="diamond"),
            hovertemplate="<b>Depth:</b> %{y:.0f} m<br><b>Predicted Anomaly:</b> %{x:+.2f} °C<extra></extra>",
            name="Predicted Anomaly ΔT(z)",
        )
    )

    # 3. Final Reconstructed Profile
    fig.add_trace(
        go.Scatter(
            x=temp_arr,
            y=depths_arr,
            mode="lines+markers",
            line=dict(color="#0284c7", width=3),
            marker=dict(
                size=7,
                color="#0369a1",
                symbol="circle",
                line=dict(color="#ffffff", width=1),
            ),
            hovertemplate="<b>Depth:</b> %{y:.0f} m<br><b>Final Reconstructed T(z):</b> %{x:.2f} °C<extra></extra>",
            name="Final Profile T(z) = T_clim + ΔT",
        )
    )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        title=dict(
            text="<b>Thermodynamic Decomposition: Prior + Anomaly = Profile</b>",
            font=dict(size=14, color="#0f172a"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text="<b>Temperature / Anomaly (°C)</b>", font=dict(size=11, color="#334155")),
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5,
            tickfont=dict(size=10, color="#475569"),
        ),
        yaxis=dict(
            title=dict(text="<b>Depth (m)</b>", font=dict(size=11, color="#334155")),
            autorange="reversed",
            showgrid=True,
            gridcolor="#e2e8f0",
            tickvals=[0, 50, 100, 150, 200, 300, 500, 700, 1000],
            tickfont=dict(size=10, color="#475569"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#334155"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=65, r=25, t=80, b=45),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=460,
    )

    return fig
