"""Regime context visualization component for OceanEmbed dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def render_regime_chart(regime_probs: list[float] | np.ndarray) -> go.Figure:
    """Render horizontal bar chart for K=4 soft latent regime context probabilities."""
    probs = np.asarray(regime_probs, dtype=np.float32)
    percentages = probs * 100.0

    regime_labels = [
        "Regime 1 (Thermal Stratification)",
        "Regime 2 (Northern River Freshening)",
        "Regime 3 (Cyclonic Eddy / Upwelling)",
        "Regime 4 (Anticyclonic Deep Mixing)",
    ]

    colors = ["#0284c7", "#0ea5e9", "#06b6d4", "#38bdf8"]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=regime_labels[: len(probs)],
            orientation="h",
            marker=dict(
                color=colors[: len(probs)],
                line=dict(color="#0369a1", width=1),
            ),
            text=[f"  {p:.1f}%" for p in percentages],
            textposition="outside",
            textfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b><br>Probability: <b>%{x:.2f}%</b><extra></extra>",
        )
    )

    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        title=dict(
            text="<b>Latent Hydrographic Regime Distribution (K = 4)</b>",
            font=dict(size=14, color="#0f172a"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="<b>Posterior Probability (%)</b>",
            range=[0, max(100.0, float(np.max(percentages)) * 1.25)],
            showgrid=True,
            gridcolor="#e2e8f0",
            tickfont=dict(size=10, color="#475569"),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#1e293b"),
        ),
        margin=dict(l=10, r=40, t=65, b=35),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=240,
    )

    return fig
