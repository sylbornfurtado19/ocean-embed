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
    lat = float(result["latitude"])
    lon = float(result["longitude"])

    # Modern 6-column KPI grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Target Location</div>
                <div class="kpi-value">{lat:.2f}°N</div>
                <div class="kpi-sub">{lon:.2f}°E · Bay of Bengal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Prediction Date</div>
                <div class="kpi-value" style="font-size: 1.15rem;">{pred_date}</div>
                <div class="kpi-sub">31-Day Sequence Center</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Surface Temp (0 m)</div>
                <div class="kpi-value">{surf_temp:.2f} <span style="font-size: 0.9rem; font-weight: 500;">°C</span></div>
                <div class="kpi-sub">1000m: {deep_temp:.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Mean Uncertainty</div>
                <div class="kpi-value">±{mean_sigma:.2f} <span style="font-size: 0.9rem; font-weight: 500;">°C</span></div>
                <div class="kpi-sub">90% Band: ±1.645σ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Val RMSE (Synth)</div>
                <div class="kpi-value">{benchmark_val_rmse:.2f} <span style="font-size: 0.9rem; font-weight: 500;">°C</span></div>
                <div class="kpi-sub">Held-out Spatial Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Inference Latency</div>
                <div class="kpi-value" style="color: #0284c7;">{inf_time:.1f} <span style="font-size: 0.9rem; font-weight: 500;">ms</span></div>
                <div class="kpi-sub">PyTorch CPU Execution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_surface_inputs(surface_inputs: dict[str, float]) -> None:
    """Render representative demo surface inputs feeding the 31-day spatiotemporal sequence."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">SST · Sea Surface Temp</div>
                <div class="sub-card-val">{surface_inputs.get('sst', 0.0):.2f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">SSS · Sea Surface Salinity</div>
                <div class="sub-card-val">{surface_inputs.get('sss', 0.0):.2f} PSU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">SLA · Sea Level Anomaly</div>
                <div class="sub-card-val">{surface_inputs.get('sla', 0.0):+.3f} m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">Wind-U · Zonal Component</div>
                <div class="sub-card-val">{surface_inputs.get('wind_u', 0.0):.2f} m/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">Wind-V · Meridional</div>
                <div class="sub-card-val">{surface_inputs.get('wind_v', 0.0):.2f} m/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_embedding_panel(embedding: np.ndarray) -> None:
    """Render compact statistics and vector inspection for the 512-D Ocean Embedding."""
    emb_arr = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb_arr))
    mean_val = float(np.mean(emb_arr))
    std_val = float(np.std(emb_arr))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">Embedding Dimensions</div>
                <div class="sub-card-val">{len(emb_arr)}-D</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">L2 Vector Norm</div>
                <div class="sub-card-val">{norm:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">Mean Activation</div>
                <div class="sub-card-val">{mean_val:+.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="sub-card">
                <div class="sub-card-label">Std Activation</div>
                <div class="sub-card-val">{std_val:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔍 Inspect 512-D Ocean Embedding Latent Vector", expanded=False):
        st.caption(
            "Continuous latent state representation generated by the ConvLSTM + CBAM attention encoder. "
            "Encapsulates 3D thermodynamic state, mixing dynamics, and thermocline structure in a 512-D bottleneck."
        )
        formatted_vec = ", ".join([f"{v:+.4f}" for v in emb_arr])
        st.code(f"// OceanEmbed V2 Latent Bottleneck Vector (512-D)\n[{formatted_vec}]", language="text")
