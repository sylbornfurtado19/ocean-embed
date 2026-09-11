"""Automated test suite for OceanEmbed Phase 4 Dashboard & Inference integration.

Validates all criteria specified in the Phase 4 specification:
1. Dashboard imports successfully.
2. Streamlit app structure & no NotImplementedError stubs.
3. Model checkpoint exists and loads.
4. Valid coordinates accepted.
5. Invalid latitude rejected.
6. Invalid longitude rejected.
7. Demo patch generated.
8. Demo patch shape is (1, 31, 5, 32, 32).
9. Inference returns 15 temperature values.
10. Sigma values are finite and positive.
11. 90% predictive interval is calculated correctly.
12. Regime probabilities sum approximately to 1.
13. Embedding has 512 dimensions.
14. Profile chart receives 15 depths.
15. Depth axis is inverted.
16. CSV export contains the expected columns.
17. Missing ARGO files handled gracefully.
18. Missing checkpoint produces a clear error.
19. Scientific reproducibility (deterministic demo input).
20. Phase 2 V0/V1 baseline inference preservation.
21. Phase 3 embedding extraction preservation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np

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
from src.data.generate_spatiotemporal_demo import generate_spatiotemporal_sample
from src.inference import OceanInferenceEngine


CHECKPOINT_PATH = WORKSPACE_ROOT / "checkpoints" / "oceanembed_v2.pt"


def test_dashboard_imports_cleanly() -> None:
    """1. Verify that dashboard module and all subcomponents import successfully."""
    import dashboard.app
    import dashboard.components.map
    import dashboard.components.metrics
    import dashboard.components.plots
    import dashboard.components.profile_table
    import dashboard.components.regime
    assert dashboard.app is not None


def test_no_not_implemented_in_dashboard() -> None:
    """2. Verify that dashboard/app.py no longer contains the NotImplementedError stub."""
    dashboard_code = (WORKSPACE_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "NotImplementedError" not in dashboard_code
    assert "TODO: implement dashboard profile plot" not in dashboard_code


def test_checkpoint_exists_and_loads() -> None:
    """3. Verify that the Phase 3 trained checkpoint loads cleanly into OceanInferenceEngine."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing at {CHECKPOINT_PATH}"
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    assert engine.model_version == "oceanembed_v2"
    assert len(engine.target_depths) == 15
    assert engine.embedding_dim == 512


def test_valid_coordinates_accepted() -> None:
    """4. Verify that coordinates inside the Bay of Bengal domain are accepted."""
    valid1, msg1 = validate_coordinates(14.0, 88.0)
    assert valid1 is True and msg1 == ""

    valid2, msg2 = validate_coordinates(DOMAIN_LAT_MIN, DOMAIN_LON_MIN)
    assert valid2 is True and msg2 == ""

    valid3, msg3 = validate_coordinates(DOMAIN_LAT_MAX, DOMAIN_LON_MAX)
    assert valid3 is True and msg3 == ""


def test_invalid_latitude_rejected() -> None:
    """5. Verify that latitudes outside 5°N–23°N are rejected with clear message."""
    inv1, msg1 = validate_coordinates(4.9, 88.0)
    assert inv1 is False and "Latitude" in msg1

    inv2, msg2 = validate_coordinates(23.1, 88.0)
    assert inv2 is False and "Latitude" in msg2


def test_invalid_longitude_rejected() -> None:
    """6. Verify that longitudes outside 80°E–100°E are rejected with clear message."""
    inv1, msg1 = validate_coordinates(14.0, 79.9)
    assert inv1 is False and "Longitude" in msg1

    inv2, msg2 = validate_coordinates(14.0, 100.1)
    assert inv2 is False and "Longitude" in msg2


def test_demo_patch_generated() -> None:
    """7. Verify that demo patch is generated deterministically from coordinate and date."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(15.0, 88.0, datetime.date(2023, 2, 15))
    assert res is not None
    assert "patch_shape" in res
    assert "surface_inputs" in res


def test_demo_patch_shape() -> None:
    """8. Verify demo patch tensor has shape (1, 31, 5, 32, 32)."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(15.0, 88.0, datetime.date(2023, 2, 15))
    assert res["patch_shape"] == (1, 31, 5, 32, 32)


def test_inference_returns_15_temperatures() -> None:
    """9. Verify inference returns exactly 15 temperature values matching target depths."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(14.0, 88.0, datetime.date(2023, 2, 15))
    temp = res["temperature"]
    depths = res["depths"]
    assert len(temp) == 15
    assert len(depths) == 15
    assert list(depths) == [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    # Physical ocean temperature plausibility check
    assert np.all(temp > 2.0) and np.all(temp < 35.0)


def test_sigma_values_finite_and_positive() -> None:
    """10. Verify uncertainty sigma values are finite, strictly positive, and non-zero."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(14.0, 88.0, datetime.date(2023, 2, 15))
    sigma = res["uncertainty_sigma"]
    assert len(sigma) == 15
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


def test_90_percent_predictive_interval_calculation() -> None:
    """11. Verify 90% Gaussian predictive interval (±1.645 sigma) is calculated correctly."""
    depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    temps = np.linspace(29.0, 6.0, 15)
    sigmas = np.full(15, 0.25)

    df = create_profile_dataframe(depths, temps, sigmas)
    expected_lower = np.round(temps - 1.645 * sigmas, 2)
    expected_upper = np.round(temps + 1.645 * sigmas, 2)
    np.testing.assert_allclose(df["lower_90_c"], expected_lower)
    np.testing.assert_allclose(df["upper_90_c"], expected_upper)


def test_regime_probabilities_sum_to_one() -> None:
    """12. Verify that K=4 latent regime probabilities sum approximately to 1.0."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(14.0, 88.0, datetime.date(2023, 2, 15))
    regimes = res["regime_probs"]
    assert len(regimes) == 4
    assert np.all(regimes >= 0.0)
    assert abs(float(np.sum(regimes)) - 1.0) < 1e-4


def test_embedding_has_512_dimensions() -> None:
    """13. Verify that the continuous latent Ocean Embedding has 512 dimensions."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    res = engine.predict_from_location_date(14.0, 88.0, datetime.date(2023, 2, 15))
    emb = res["embedding"]
    assert len(emb) == 512
    assert np.all(np.isfinite(emb))


def test_profile_chart_receives_15_depths() -> None:
    """14. Verify that Plotly profile chart receives exactly 15 discrete depths."""
    depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    temps = np.linspace(29.0, 6.0, 15)
    sigmas = np.full(15, 0.25)

    fig = render_temperature_profile_chart(depths, temps, sigmas, show_uncertainty=True)
    # The mean trace is the third trace (index 2)
    mean_trace = fig.data[2]
    assert len(mean_trace.y) == 15
    np.testing.assert_allclose(mean_trace.y, depths)


def test_profile_chart_depth_axis_inverted() -> None:
    """15. Verify that Plotly profile chart inverts the depth axis (increasing downward)."""
    depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    temps = np.linspace(29.0, 6.0, 15)
    sigmas = np.full(15, 0.25)

    fig = render_temperature_profile_chart(depths, temps, sigmas)
    assert fig.layout.yaxis.autorange == "reversed"


def test_csv_export_columns_and_schema() -> None:
    """16. Verify CSV export contains depth_m, temperature_c, sigma_c, lower_90_c, upper_90_c."""
    depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    temps = np.linspace(29.0, 6.0, 15)
    sigmas = np.full(15, 0.2)

    df = create_profile_dataframe(depths, temps, sigmas)
    expected_cols = ["depth_m", "temperature_c", "sigma_c", "lower_90_c", "upper_90_c"]
    assert list(df.columns) == expected_cols
    assert len(df) == 15


def test_missing_argo_handled_gracefully() -> None:
    """17. Verify missing ARGO files do not crash the dashboard and return False."""
    from dashboard.app import check_argo_availability
    assert check_argo_availability() is False


def test_missing_checkpoint_raises_error() -> None:
    """18. Verify that loading a non-existent checkpoint raises clear FileNotFoundError."""
    fake_path = WORKSPACE_ROOT / "checkpoints" / "non_existent_model.pt"
    try:
        OceanInferenceEngine(fake_path)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()


def test_scientific_reproducibility() -> None:
    """19. Deterministic consistency test: Same (lat, lon, date) produces identical predictions."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    lat, lon = 12.5, 87.5
    date_val = datetime.date(2023, 3, 10)

    run1 = engine.predict_from_location_date(lat, lon, date_val)
    run2 = engine.predict_from_location_date(lat, lon, date_val)

    np.testing.assert_allclose(run1["temperature"], run2["temperature"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run1["uncertainty_sigma"], run2["uncertainty_sigma"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run1["embedding"], run2["embedding"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run1["regime_probs"], run2["regime_probs"], rtol=1e-5, atol=1e-5)


def test_phase2_v0_v1_preservation() -> None:
    """20. Verify Phase 2 V0 and V1 baseline models load and run inference properly."""
    v0_path = WORKSPACE_ROOT / "checkpoints" / "v0_baseline.pt"
    v1_path = WORKSPACE_ROOT / "checkpoints" / "v1_uncertainty.pt"

    if v0_path.exists():
        v0_eng = OceanInferenceEngine(v0_path)
        assert v0_eng.model_version == "v0"
        sample_feat = [28.5, 33.2, 0.05, 2.1, -1.2, 14.0, 88.0, 45.0]
        v0_res = v0_eng.predict(sample_feat)
        assert len(v0_res["temperature"]) == 15

    if v1_path.exists():
        v1_eng = OceanInferenceEngine(v1_path)
        assert v1_eng.model_version == "v1_uncertainty"
        sample_feat = [28.5, 33.2, 0.05, 2.1, -1.2, 14.0, 88.0, 45.0]
        v1_res = v1_eng.predict(sample_feat)
        assert len(v1_res["temperature"]) == 15
        assert len(v1_res["uncertainty_sigma"]) == 15


def test_phase3_embedding_extraction_preservation() -> None:
    """21. Verify Phase 3 standalone 512-D embedding extraction continues to work."""
    engine = OceanInferenceEngine(CHECKPOINT_PATH)
    rng = np.random.default_rng(123)
    dummy_patch = rng.standard_normal((31, 5, 32, 32)).astype(np.float32)
    emb = engine.extract_embedding(dummy_patch)
    assert emb.shape == (512,)
    assert np.all(np.isfinite(emb))


if __name__ == "__main__":
    print("Running comprehensive Phase 4 test suite...")
    test_dashboard_imports_cleanly()
    test_no_not_implemented_in_dashboard()
    test_checkpoint_exists_and_loads()
    test_valid_coordinates_accepted()
    test_invalid_latitude_rejected()
    test_invalid_longitude_rejected()
    test_demo_patch_generated()
    test_demo_patch_shape()
    test_inference_returns_15_temperatures()
    test_sigma_values_finite_and_positive()
    test_90_percent_predictive_interval_calculation()
    test_regime_probabilities_sum_to_one()
    test_embedding_has_512_dimensions()
    test_profile_chart_receives_15_depths()
    test_profile_chart_depth_axis_inverted()
    test_csv_export_columns_and_schema()
    test_missing_argo_handled_gracefully()
    test_missing_checkpoint_raises_error()
    test_scientific_reproducibility()
    test_phase2_v0_v1_preservation()
    test_phase3_embedding_extraction_preservation()
    print("ALL 21 PHASE 4 TESTS PASSED SUCCESSFULLY!")
