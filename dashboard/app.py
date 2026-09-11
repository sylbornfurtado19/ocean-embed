"""OceanEmbed — Subsurface Ocean Temperature Reconstruction Dashboard.

Smart India Hackathon (SIH 2026) | Problem Statement 26066
Team: Bug Dealers | Category: Software | Theme: Disaster Management
Domain: Bay of Bengal (5°N–23°N, 80°E–100°E) | 15 Target Depths (0 to 1000 m)
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import streamlit as st

from dashboard.components.map import (
    DOMAIN_LAT_MAX,
    DOMAIN_LAT_MIN,
    DOMAIN_LON_MAX,
    DOMAIN_LON_MIN,
    render_bay_of_bengal_map,
    validate_coordinates,
)
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
from src.inference import OceanInferenceEngine

# Page Configuration
st.set_page_config(
    page_title="OceanEmbed | Subsurface Ocean Reconstruction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Oceanography & Disaster Management UI Design System
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    /* Header & Hero Section */
    .hero-header {
        background: linear-gradient(135deg, #0a192f 0%, #0f2b48 50%, #0c4a6e 100%);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(10, 25, 47, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.4;
    }
    .badge-group {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .badge-sih {
        background: rgba(56, 189, 248, 0.15);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-mode-demo {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.45);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-mode-real {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Scientific Disclaimer Notice Card */
    .disclaimer-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        padding: 0.85rem 1.15rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    /* Professional Card Containers */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .section-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* KPI Metrics Card Grid */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 9px;
        padding: 0.85rem 1.0rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Sub-cards for surface conditions & embeddings */
    .sub-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .sub-card-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .sub-card-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Thermodynamic Equation Banner */
    .equation-banner {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 8px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        border-left: 4px solid #38bdf8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    /* In-situ ARGO Status Card */
    .argo-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-top: 0.75rem;
    }
    .argo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .status-pill-unavail {
        background: #fef3c7;
        color: #b45309;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #fde68a;
    }
    .status-pill-avail {
        background: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #bbf7d0;
    }

    /* Primary Action Button Polish */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        padding: 0.65rem 1.25rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 4px 10px rgba(2, 132, 199, 0.35);
        transform: translateY(-1px);
    }

    /* Footer Styling */
    .footer-container {
        border-top: 1px solid #e2e8f0;
        padding-top: 1.25rem;
        margin-top: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        color: #64748b;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading OceanEmbed V2 checkpoint into memory...")
def load_cached_inference_engine(checkpoint_path: str = "checkpoints/oceanembed_v2.pt") -> OceanInferenceEngine:
    """Load and cache the trained OceanEmbed V2 model on CPU."""
    ckpt_file = WORKSPACE_ROOT / checkpoint_path
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"OceanEmbed V2 checkpoint not found at: {ckpt_file}. "
            "Please train Phase 3 model first using 'python src/train_oceanembed.py'."
        )
    return OceanInferenceEngine(ckpt_file)


def check_argo_availability() -> bool:
    """Check if real ARGO float NetCDF files exist in the data directory."""
    argo_dir = WORKSPACE_ROOT / "data" / "raw" / "argo"
    if not argo_dir.exists():
        return False
    nc_files = list(argo_dir.glob("*.nc")) + list(argo_dir.glob("*.nc4")) + list(argo_dir.glob("**/*.nc"))
    return len(nc_files) > 0


def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "lat" not in st.session_state:
        st.session_state["lat"] = 14.00
    if "lon" not in st.session_state:
        st.session_state["lon"] = 88.00
    if "date" not in st.session_state:
        st.session_state["date"] = datetime.date(2023, 2, 15)
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None


def main() -> None:
    """Render the OceanEmbed scientific interactive dashboard."""
    init_session_state()

    # Model Engine Loading with Safe Error Handling
    try:
        engine = load_cached_inference_engine()
    except FileNotFoundError as e:
        st.error(f"❌ Checkpoint Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load OceanEmbed model: {e}")
        st.stop()

    # Sidebar: Clean Prediction Controls Panel
    with st.sidebar:
        st.markdown("### 🎛️ Prediction Controls")
        st.caption("Bay of Bengal Prototype (5.0°N–23.0°N, 80.0°E–100.0°E)")

        input_lat = st.number_input(
            "Target Latitude (°N)",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state["lat"]),
            step=0.25,
            format="%.2f",
            help="Target latitude within the Bay of Bengal domain (5°N to 23°N)",
        )
        input_lon = st.number_input(
            "Target Longitude (°E)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state["lon"]),
            step=0.25,
            format="%.2f",
            help="Target longitude within the Bay of Bengal domain (80°E to 100°E)",
        )
        input_date = st.date_input(
            "Prediction Date",
            value=st.session_state["date"],
            min_value=datetime.date(2020, 1, 1),
            max_value=datetime.date(2026, 12, 31),
            help="Central date for the 31-day temporal sequence (day -15 to day +15)",
        )

        st.markdown("---")
        st.markdown("### 📡 Observation Feed")
        sat_dir = WORKSPACE_ROOT / "data" / "raw" / "satellite"
        real_sat_available = sat_dir.exists() and len(list(sat_dir.glob("*.nc")) + list(sat_dir.glob("*.zarr"))) > 0
        data_source_options = ["Synthetic Demo (Active)"]
        if real_sat_available:
            data_source_options.append("Real Satellite (CMEMS)")
        else:
            data_source_options.append("Real Satellite (Not Configured)")

        selected_data_source = st.radio(
            "Input Surface Data Mode",
            data_source_options,
            index=0,
            help="Select verified deterministic synthetic simulation or ingested real satellite rasters.",
        )
        is_real_data_mode = (selected_data_source == "Real Satellite (CMEMS)")

        st.markdown("---")
        st.markdown("### ⚙️ Analytical Layers")
        show_uncertainty = st.checkbox("90% Gaussian Uncertainty Ribbon (±1.645σ)", value=True)
        show_climatology = st.checkbox("Climatology Residual Decomposition", value=True)
        show_regime = st.checkbox("Latent Regime Probabilities (K = 4)", value=True)
        show_surface = st.checkbox("Surface Boundary Inputs", value=True)
        show_embedding = st.checkbox("512-D Ocean Embedding State", value=True)
        show_argo = st.checkbox("In-Situ ARGO Validation Layer", value=True)

        st.markdown("---")
        reconstruct_clicked = st.button(
            "🌊 RECONSTRUCT PROFILE",
            type="primary",
            use_container_width=True,
            help="Generate 31-day spatiotemporal sequence and execute OceanEmbed V2 forward pass",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
                <b>Model:</b> OceanEmbed V2 (PyTorch CPU)<br>
                <b>Encoder:</b> 2D CNN + ConvLSTM + CBAM<br>
                <b>Latent Bottleneck:</b> 512 Dimensions<br>
                <b>Decoder:</b> Depth-Conditioned Continuous MLP<br>
                <b>Target Depths:</b> 15 Levels (0 to 1000 m)<br>
                <b>Domain:</b> Bay of Bengal Basin
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Header / Hero Section
    badge_html = (
        '<span class="badge-mode-real">● REAL SATELLITE (SYNTHETIC WEIGHTS)</span>'
        if is_real_data_mode
        else '<span class="badge-mode-demo">● DEMO MODE — SYNTHETIC DATA</span>'
    )

    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title-row">
                <h1 class="hero-title">🌊 OceanEmbed</h1>
                <div class="badge-group">
                    <span class="badge-sih">SIH 2026 · PS 26066</span>
                    {badge_html}
                </div>
            </div>
            <p class="hero-subtitle">
                Satellite-Embedding Deep Learning Framework for Subsurface Ocean Thermal Structure Reconstruction
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Disclaimer Notice
    st.markdown(
        """
        <div class="disclaimer-card">
            <span style="font-size: 1.2rem;">🔬</span>
            <div>
                <b>Scientific Operational Notice:</b> Predictions are generated using deterministic synthetic development data to verify software architecture and model mechanics. 
                They do not represent operational operational forecasts, GLORYS reanalysis, or live ARGO measurements.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Grid: Left Column = Folium Map, Right Column = Domain Guide & Quick Status
    col_map, col_status = st.columns([1.35, 1.0])

    with col_map:
        st.markdown("##### 📍 Geographic Domain Selector (Bay of Bengal)")
        st.caption("Click any point within the highlighted rectangular domain to set reconstruction target coordinates:")
        clicked_lat, clicked_lon = render_bay_of_bengal_map(input_lat, input_lon)
        if clicked_lat is not None and clicked_lon is not None:
            is_valid, _ = validate_coordinates(clicked_lat, clicked_lon)
            if is_valid:
                if round(clicked_lat, 2) != round(float(st.session_state["lat"]), 2) or round(clicked_lon, 2) != round(float(st.session_state["lon"]), 2):
                    st.session_state["lat"] = clicked_lat
                    st.session_state["lon"] = clicked_lon
                    st.rerun()

    with col_status:
        st.markdown("##### 🧭 Target Parameters & Domain Boundary")
        is_coord_valid, coord_err = validate_coordinates(input_lat, input_lon)
        if not is_coord_valid:
            st.error(f"⚠️ {coord_err}")
            st.warning("Coordinates are outside the supported Bay of Bengal operational prototype (5°N–23°N, 80°E–100°E).")
        else:
            st.markdown(
                f"""
                <div class="section-card" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 0.85rem; color: #0f172a;">COORDINATE STATUS</span>
                        <span style="color: #16a34a; font-weight: 700; font-size: 0.78rem; background: #dcfce7; padding: 2px 8px; border-radius: 4px;">VALID DOMAIN</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{input_lat:.4f}°N, {input_lon:.4f}°E</div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.2rem;">Bay of Bengal Basin · 0.25° Resolution Grid</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            start_t = input_date - datetime.timedelta(days=15)
            end_t = input_date + datetime.timedelta(days=15)
            st.markdown(
                f"""
                <div class="sub-card" style="margin-bottom: 0.75rem;">
                    <div class="sub-card-label">Temporal Window (31 Days)</div>
                    <div style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">{input_date.isoformat()} (Center: Day 0)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Sequence: {start_t.isoformat()} to {end_t.isoformat()}</div>
                </div>
                <div class="sub-card" style="margin-bottom: 0;">
                    <div class="sub-card-label">Spatial Neighborhood</div>
                    <div style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">32 × 32 Patch (~8° × 8°)</div>
                    <div style="font-size: 0.75rem; color: #64748b;">5 Surface Channels: SST · SSS · SLA · Wind-U · Wind-V</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Inference Execution
    if reconstruct_clicked:
        if not is_coord_valid:
            st.error("Cannot reconstruct profile: Selected location is outside the supported domain.")
        else:
            st.session_state["lat"] = input_lat
            st.session_state["lon"] = input_lon
            st.session_state["date"] = input_date

            with st.spinner("Executing OceanEmbed V2 forward pass..."):
                try:
                    res = engine.predict_from_location_date(input_lat, input_lon, input_date)
                    st.session_state["prediction_result"] = res
                except Exception as ex:
                    st.error(f"Inference forward pass failed: {ex}")
                    st.session_state["prediction_result"] = None

    result = st.session_state.get("prediction_result", None)

    # Results Display
    if result is not None:
        st.markdown("---")
        st.markdown("### 📊 Reconstruction Summary Metrics")
        render_summary_metrics(result)

        st.markdown("---")
        st.markdown("### 🌡️ Vertical Ocean Temperature Profile")

        # Vertical Profile Chart
        fig_profile = render_temperature_profile_chart(
            depths=result["depths"],
            temperature=result["temperature"],
            sigma=result["uncertainty_sigma"],
            show_uncertainty=show_uncertainty,
        )
        st.plotly_chart(fig_profile, use_container_width=True)

        # Climatology Decomposition (Expander or Inline)
        if show_climatology and "climatology_prior" in result and "anomaly" in result:
            with st.expander("🔍 Climatology Residual Decomposition Analysis", expanded=False):
                st.markdown(
                    """
                    <div class="equation-banner">
                        <span>Physical Formulation: <b>T(z) = T<sub>climatology</sub>(z) + ΔT(z)</b></span>
                        <span style="font-size: 0.8rem; color: #7dd3fc;">Prior-Guided Deep Ocean Stabilization</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_clim = render_climatology_decomposition_chart(
                    depths=result["depths"],
                    temperature=result["temperature"],
                    climatology=result["climatology_prior"],
                    anomaly=result["anomaly"],
                )
                st.plotly_chart(fig_clim, use_container_width=True)

        # Side-by-side: Regime Context & Surface Conditions
        col_regime, col_surf = st.columns(2)

        with col_regime:
            if show_regime and "regime_probs" in result:
                st.markdown("##### 🌐 Hydrographic Regime Context")
                fig_regime = render_regime_chart(result["regime_probs"])
                st.plotly_chart(fig_regime, use_container_width=True)

        with col_surf:
            if show_surface and "surface_inputs" in result:
                st.markdown("##### 🌊 Surface Boundary Conditions (Center Date)")
                render_surface_inputs(result["surface_inputs"])

        # Profile Data Table & CSV Download
        with st.expander("📋 Profile Data Table & Export (15 Standard Depths)", expanded=True):
            st.caption(
                "Discrete 15-depth vertical temperature reconstruction with predictive standard deviation (σ) "
                "and 90% Gaussian predictive confidence intervals (±1.645σ)."
            )
            profile_df = create_profile_dataframe(
                depths=result["depths"],
                temperature=result["temperature"],
                sigma=result["uncertainty_sigma"],
            )
            render_profile_table(
                profile_df,
                lat=result["latitude"],
                lon=result["longitude"],
                date_str=result["prediction_date"],
            )

        # 512-D Ocean Embedding Inspection
        if show_embedding and "embedding" in result:
            st.markdown("---")
            st.markdown("### 🧬 512-D Ocean Embedding (Latent State Representation)")
            render_embedding_panel(result["embedding"])

        # ARGO Validation Status
        if show_argo:
            st.markdown("---")
            st.markdown("### 🎯 In-Situ ARGO Float Verification")
            argo_available = check_argo_availability()
            if not argo_available:
                st.markdown(
                    """
                    <div class="argo-card">
                        <div class="argo-header">
                            <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Independent In-Situ Validation Status</span>
                            <span class="status-pill-unavail">STATUS: NOT AVAILABLE</span>
                        </div>
                        <p style="margin-top: 0.35rem; margin-bottom: 0.25rem; color: #334155; font-size: 0.88rem;">
                            <b>ARGO float comparison unavailable</b> — No authentic NetCDF float observation profiles are ingested in <code>data/raw/argo</code>.
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.80rem;">
                            To maintain absolute scientific honesty, the evaluation framework gracefully skips in-situ verification without fabricating simulated float data. Real ARGO NetCDF files can be downloaded using <code>python scripts/download_argo.py</code>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                from src.api.services.argo_service import ArgoService
                argo_srv = ArgoService()
                argo_status = argo_srv.get_argo_status()
                st.markdown(
                    f"""
                    <div class="argo-card">
                        <div class="argo-header">
                            <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Independent In-Situ Validation Status</span>
                            <span class="status-pill-avail">STATUS: ACTIVE ({argo_status['profiles_available']} PROFILES)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                nearby = argo_srv.get_profiles(result["latitude"], result["longitude"], radius_deg=2.5)
                if nearby:
                    nearest = nearby[0]
                    st.info(
                        f"📍 **Nearest Float Profile:** `{nearest['profile_id']}` ({nearest['latitude']}°N, {nearest['longitude']}°E) · "
                        f"Depth Range: {nearest['depth_min']} m to {nearest['depth_max']} m ({nearest['num_levels']} levels)"
                    )
                else:
                    st.caption("No ARGO float profiles located within 2.5° radius of selected target coordinate.")

    else:
        st.info("👈 Select your desired coordinates and prediction date in the sidebar or map, then click **RECONSTRUCT PROFILE** to run the OceanEmbed V2 model.")

    # Technical Specifications Footer
    st.markdown(
        """
        <div class="footer-container">
            <div>
                <b>OceanEmbed</b> · Smart India Hackathon 2026 (Problem Statement 26066) · Team Bug Dealers
            </div>
            <div>
                Architecture: ConvLSTM + CBAM + 512-D Latent Bottleneck + Depth-Conditioned Decoder
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
