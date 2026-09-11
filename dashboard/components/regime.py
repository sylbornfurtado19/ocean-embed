"""Regime context visualization component for OceanEmbed dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def render_regime_chart(regime_probs: list[float] | np.ndarray) -> go.Figure:
    """Render horizontal bar chart for K=4 soft latent regime context probabilities."""
    probs = np.asarray(regime_probs, dtype=np.float32)
    percentages = probs * 100.0

    regime_labels = [
        "Latent Regime 1",
        "Latent Regime 2",
        "Latent Regime 3",
        "Latent Regime 4",
    ]

    colors = ["#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc"]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=regime_labels,
            orientation="h",
            marker=dict(
                color=colors[: len(probs)],
                line=dict(color="#0369a1", width=1),
            ),
            text=[f"{p:.1f}%" for p in percentages],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=11, family="sans-serif"),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Latent Regime Distribution (K = 4)</b><br><sup>Soft differentiable regime probabilities learned by OceanEmbed</sup>",
            font=dict(size=14, color="#0f172a"),
            x=0.02,
        ),
        xaxis=dict(
            title="<b>Probability (%)</b>",
            range=[0, 100],
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=10, color="#475569"),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#334155"),
        ),
        margin=dict(l=10, r=20, t=65, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=240,
    )

    return fig
