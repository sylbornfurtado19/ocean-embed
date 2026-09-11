"""Summary metrics, surface inputs, and latent embedding components for OceanEmbed dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st


def render_summary_metrics(
    result: dict[str, Any],
    benchmark_val_rmse: float = 0.2559,
) -> None:
    """Render compact, professional KPI cards for the reconstruction run."""
    depths = result["depths"]
    temps = result["temperature"]
    sigmas = result["uncertainty_sigma"]

    surf_temp = float(temps[0])
    deep_temp = float(temps[-1])
    mean_sigma = float(np.mean(sigmas))
    inf_time = float(result.get("inference_time_ms", 0.0))
    pred_date = str(result.get("prediction_date", "N/A"))

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            label="Selected Location",
            value=f"{result['latitude']:.2f}°N",
            delta=f"{result['longitude']:.2f}°E",
            delta_color="off",
            help=f"Target coordinates: Latitude {result['latitude']:.4f}°N, Longitude {result['longitude']:.4f}°E",
        )
    with col2:
        st.metric(
            label="Prediction Date",
            value=pred_date,
            help="Central prediction date (day 0 of the 31-day temporal sequence)",
        )
    with col3:
        st.metric(
            label="Surface Temp (0 m)",
            value=f"{surf_temp:.2f} °C",
            delta=f"1000m: {deep_temp:.2f} °C",
            delta_color="off",
            help="Estimated temperature at the sea surface (0 m) and sea bottom (1000 m)",
        )
    with col4:
        st.metric(
            label="Mean Uncertainty (σ)",
            value=f"±{mean_sigma:.2f} °C",
            help="Average predictive standard deviation across all 15 depth levels",
        )
    with col5:
        st.metric(
            label="Validation RMSE (Synth)",
            value=f"{benchmark_val_rmse:.2f} °C",
            help="Benchmark root mean squared error on held-out synthetic validation set (not a live measurement error)",
        )
    with col6:
        st.metric(
            label="Inference Latency",
            value=f"{inf_time:.1f} ms",
            help="Measured PyTorch CPU model forward pass execution time",
        )


def render_surface_inputs(surface_inputs: dict[str, float]) -> None:
    """Render representative demo surface inputs feeding the 31-day spatiotemporal sequence."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="SST (Sea Surface Temp)",
            value=f"{surface_inputs.get('sst', 0.0):.2f} °C",
        )
    with col2:
        st.metric(
            label="SSS (Sea Surface Salinity)",
            value=f"{surface_inputs.get('sss', 0.0):.2f} PSU",
        )
    with col3:
        st.metric(
            label="SLA (Sea Level Anomaly)",
            value=f"{surface_inputs.get('sla', 0.0):+.3f} m",
        )
    with col4:
        st.metric(
            label="Zonal Wind (U)",
            value=f"{surface_inputs.get('wind_u', 0.0):.2f} m/s",
        )
    with col5:
        st.metric(
            label="Meridional Wind (V)",
            value=f"{surface_inputs.get('wind_v', 0.0):.2f} m/s",
        )


def render_embedding_panel(embedding: np.ndarray) -> None:
    """Render compact statistics and vector inspection for the 512-D Ocean Embedding."""
    emb_arr = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb_arr))
    mean_val = float(np.mean(emb_arr))
    std_val = float(np.std(emb_arr))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Embedding Dimensionality", f"{len(emb_arr)}-D")
    with c2:
        st.metric("L2 Norm", f"{norm:.3f}")
    with c3:
        st.metric("Mean Activation", f"{mean_val:.4f}")
    with c4:
        st.metric("Std Activation", f"{std_val:.4f}")

    with st.expander("🔍 View 512-D Ocean Embedding Vector", expanded=False):
        st.caption(
            "Learned compact latent ocean-state representation output by the ConvLSTM + CBAM attention encoder. "
            "Downstream decoder and regime context head decode this vector into 15 depth temperatures."
        )
        # Display as a scrollable array representation
        formatted_vec = ", ".join([f"{v:.4f}" for v in emb_arr])
        st.code(f"[{formatted_vec}]", language="text")
