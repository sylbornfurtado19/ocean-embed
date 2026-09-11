"""OceanEmbed — Subsurface Ocean Temperature Reconstruction Dashboard.

Smart India Hackathon (SIH 2026) | Problem Statement 26066
Team: Bug Dealers | Category: Software | Theme: Disaster Management
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
    page_title="OceanEmbed | Subsurface Reconstruction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .demo-badge {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        border: 1px solid #fde68a;
    }
    .disclaimer-card {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }
    .argo-card {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
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
    nc_files = list(argo_dir.glob("*.nc"))
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
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load OceanEmbed model: {e}")
        st.stop()

    # Sidebar: Controls & Options
    with st.sidebar:
        st.markdown("### 🎛️ Prediction Controls")
        st.caption("Bay of Bengal Prototype (5°N–23°N, 80°E–100°E)")

        input_lat = st.number_input(
            "Latitude (°N)",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state["lat"]),
            step=0.25,
            format="%.2f",
            help="Target latitude within the Bay of Bengal domain (5°N to 23°N)",
        )
        input_lon = st.number_input(
            "Longitude (°E)",
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
            min_value=datetime.date(2023, 1, 1),
            max_value=datetime.date(2023, 12, 31),
            help="Central date for the 31-day temporal sequence (day -15 to day +15)",
        )

        st.markdown("---")
        st.markdown("### 📡 Data Source")
        sat_dir = WORKSPACE_ROOT / "data" / "raw" / "satellite"
        real_sat_available = sat_dir.exists() and len(list(sat_dir.glob("*.nc")) + list(sat_dir.glob("*.zarr"))) > 0
        data_source_options = ["Synthetic Demo (Active)"]
        if real_sat_available:
            data_source_options.append("Real Satellite (CMEMS)")
        else:
            data_source_options.append("Real Satellite (Not Configured)")

        selected_data_source = st.radio(
            "Select Input Source",
            data_source_options,
            index=0,
            help="Select verified deterministic synthetic simulation or ingested real satellite rasters.",
        )
        is_real_data_mode = (selected_data_source == "Real Satellite (CMEMS)")

        st.markdown("---")
        st.markdown("### ⚙️ View Options")
        show_uncertainty = st.checkbox("Show 90% Uncertainty Envelope", value=True)
        show_climatology = st.checkbox("Show Climatology Decomposition", value=True)
        show_regime = st.checkbox("Show Latent Regime Context", value=True)
        show_surface = st.checkbox("Show Surface Inputs", value=True)
        show_embedding = st.checkbox("Show 512-D Ocean Embedding", value=True)
        show_argo = st.checkbox("Show ARGO Verification Status", value=True)

        st.markdown("---")
        reconstruct_clicked = st.button(
            "🌊 RECONSTRUCT PROFILE",
            type="primary",
            use_container_width=True,
            help="Generate 31-day spatiotemporal demo patch and run OceanEmbed V2 inference",
        )

        st.markdown("---")
        st.markdown(
            """
            **Model:** OceanEmbed V2  
            **Encoder:** CNN + ConvLSTM + CBAM  
            **Latent Bottleneck:** 512-D  
            **Decoder:** Depth-Conditioned  
            **Target Depths:** 15 (0 to 1000 m)  
            **Device:** CPU  
            """
        )

    # Header with Honest Mode Badge
    badge_html = (
        '<span class="demo-badge" style="background-color: #dbeafe; color: #1e40af; border-color: #bfdbfe;">'
        'REAL SATELLITE INPUT — MODEL TRAINED ON SYNTHETIC DEVELOPMENT DATA</span>'
        if is_real_data_mode
        else '<span class="demo-badge">DEMO MODE — SYNTHETIC DATA</span>'
    )

    st.markdown(
        f"""
        <div class="main-header">
            {badge_html}
            <h1 style="margin: 0.3rem 0 0.1rem 0; font-size: 2.2rem; color: #0f172a;">OceanEmbed</h1>
            <p style="margin: 0; color: #64748b; font-size: 1.05rem;">
                Satellite-Based Subsurface Ocean Temperature Reconstruction · Smart India Hackathon 2026 (Team Bug Dealers)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Disclaimer Notice
    st.markdown(
        """
        <div class="disclaimer-card">
            <b>🔬 Scientific Notice:</b> Current predictions are generated from deterministic synthetic development data. 
            They do not represent live satellite observations, GLORYS reanalysis, or ARGO measurements and must not be interpreted as real-world ocean forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Grid: Left Column = Map, Right Column = Instructions / Quick Status
    col_map, col_status = st.columns([1.3, 1.0])

    with col_map:
        st.markdown("##### 📍 Bay of Bengal Geographic Selector")
        st.caption("Click any location inside the highlighted box to select target coordinates:")
        clicked_lat, clicked_lon = render_bay_of_bengal_map(input_lat, input_lon)
        if clicked_lat is not None and clicked_lon is not None:
            # Update coordinates if clicked point is valid
            is_valid, _ = validate_coordinates(clicked_lat, clicked_lon)
            if is_valid:
                if round(clicked_lat, 2) != round(float(st.session_state["lat"]), 2) or round(clicked_lon, 2) != round(float(st.session_state["lon"]), 2):
                    st.session_state["lat"] = clicked_lat
                    st.session_state["lon"] = clicked_lon
                    st.rerun()

    with col_status:
        st.markdown("##### 🧭 Target Parameters")
        is_coord_valid, coord_err = validate_coordinates(input_lat, input_lon)
        if not is_coord_valid:
            st.error(f"⚠️ {coord_err}")
            st.warning("Selected location is outside the current OceanEmbed demo domain. Please select coordinates within 5°N–23°N, 80°E–100°E.")
        else:
            st.success(
                f"**Latitude:** {input_lat:.4f}°N  \n"
                f"**Longitude:** {input_lon:.4f}°E"
            )
            st.info(
                f"**Prediction Date:** {input_date.isoformat()}  \n"
                f"**Temporal Window:** {input_date - datetime.timedelta(days=15)} to {input_date + datetime.timedelta(days=15)} (31 days)  \n"
                f"**Spatial Patch:** 32 × 32 (~8° × 8° local neighborhood)"
            )
            st.markdown(
                """
                **How to use:**
                1. Select location via the map or sidebar inputs.
                2. Select prediction date.
                3. Click **RECONSTRUCT PROFILE** to execute OceanEmbed V2.
                """
            )

    # Inference Execution
    if reconstruct_clicked:
        if not is_coord_valid:
            st.error("Cannot reconstruct profile: Selected location is outside the supported domain.")
        else:
            st.session_state["lat"] = input_lat
            st.session_state["lon"] = input_lon
            st.session_state["date"] = input_date

            with st.spinner("Reconstructing subsurface temperature profile via OceanEmbed V2..."):
                try:
                    res = engine.predict_from_location_date(input_lat, input_lon, input_date)
                    st.session_state["prediction_result"] = res
                except Exception as ex:
                    st.error(f"Inference failed: {ex}")
                    st.session_state["prediction_result"] = None

    result = st.session_state.get("prediction_result", None)

    # Results Display
    if result is not None:
        st.markdown("---")
        st.markdown("### 📊 Reconstruction Summary")
        render_summary_metrics(result)

        st.markdown("---")
        st.markdown("### 🌡️ Subsurface Ocean Temperature Profile")

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
            with st.expander("🔍 View Climatology Residual Decomposition", expanded=False):
                st.caption(
                    "OceanEmbed V2 learns to predict a subsurface temperature anomaly ΔT(z), which is recombined "
                    "with the synthetic demo climatology prior: Final T(z) = Climatology Prior(z) + ΔT(z)."
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
                st.markdown("##### 🌐 Latent Regime Context")
                fig_regime = render_regime_chart(result["regime_probs"])
                st.plotly_chart(fig_regime, use_container_width=True)
                st.caption("Soft latent regime representation learned by the model (differentiable probability distribution across K=4 regimes).")

        with col_surf:
            if show_surface and "surface_inputs" in result:
                st.markdown("##### 🌊 Synthetic Demo Surface Conditions")
                st.caption("Representative surface boundary conditions at the center coordinate and prediction date:")
                render_surface_inputs(result["surface_inputs"])

        # Profile Data Table & CSV Download
        with st.expander("📋 View Profile Data Table & CSV Export", expanded=True):
            st.caption(
                "Tabular representation of the 15-depth reconstructed profile showing mean temperature, "
                "predictive uncertainty (σ), and 90% Gaussian predictive bounds."
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
            st.markdown("### 🧬 Ocean Embedding (512-D Latent State)")
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
                        <span style="font-weight: 700; color: #b45309; background: #fef3c7; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">
                            STATUS: NOT AVAILABLE
                        </span>
                        <p style="margin-top: 0.5rem; margin-bottom: 0.2rem; color: #334155; font-size: 0.9rem;">
                            <b>ARGO comparison unavailable</b> — no real ARGO NetCDF profiles are currently ingested in <code>data/raw/argo</code>.
                        </p>
                        <p style="margin: 0; color: #64748b; font-size: 0.8rem;">
                            The evaluation framework gracefully skips in-situ verification to avoid fabricating unverified observations. In-situ ARGO float profiles will be integrated in subsequent operational phases.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                from src.api.services.argo_service import ArgoService
                argo_srv = ArgoService()
                argo_status = argo_srv.get_argo_status()
                st.success(f"✅ In-situ ARGO observations detected ({argo_status['profiles_available']} valid profiles in Bay of Bengal domain).")
                nearby = argo_srv.get_profiles(result["latitude"], result["longitude"], radius_deg=2.5)
                if nearby:
                    nearest = nearby[0]
                    st.info(
                        f"📍 **Nearest ARGO Profile:** `{nearest['profile_id']}`  \n"
                        f"**Coordinates:** {nearest['latitude']}°N, {nearest['longitude']}°E  \n"
                        f"**Observed Depth Range:** {nearest['depth_min']} m to {nearest['depth_max']} m ({nearest['num_levels']} levels)"
                    )
                else:
                    st.caption("No ARGO float profiles located within 2.5° radius of selected target coordinate.")

    else:
        # Prompt user to click Reconstruct Profile
        st.info("👈 Select your desired coordinates and date in the sidebar or map, then click **RECONSTRUCT PROFILE** to run the OceanEmbed V2 model.")

    # Technical Specifications Footer
    st.markdown("---")
    with st.expander("ℹ️ About the Model Architecture", expanded=False):
        st.markdown(
            """
            **OceanEmbed V2 Architecture Specification**:
            - **Input Representation:** 31-day temporal sequence × 5 surface channels (SST, SSS, SLA, Wind-U, Wind-V) × 32×32 spatial patch (~0.25° grid).
            - **Encoder:** 2D CNN feature extractor + Spatiotemporal ConvLSTM with CBAM (Convolutional Block Attention Module: Channel & Spatial Attention).
            - **Bottleneck:** 512-dimensional continuous latent Ocean Embedding.
            - **Regime Context:** K = 4 soft latent regime probability distribution.
            - **Climatology Integration:** Residual formulation: $T(z) = \\text{Climatology Prior}(z) + \\Delta T(z)$.
            - **Decoder:** Depth-Conditioned continuous MLP decoding 15 coupled depths simultaneously.
            - **Uncertainty Formulation:** Gaussian negative log-likelihood predictive variance $\\sigma^2(z)$, yielding a 90% Gaussian predictive interval ($\\pm 1.645\\sigma$).
            - **Official Depths (m):** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000.
            """
        )


if __name__ == "__main__":
    main()
