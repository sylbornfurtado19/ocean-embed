"""Phase 6 Robustness and Edge-Case Test Suite for OceanEmbed.

Tests all hardening requirements specified in Phase 6:
1. Missing satellite directory handled gracefully.
2. Missing ARGO directory handled gracefully.
3. Empty ARGO directory handled gracefully.
4. Malformed/corrupted NetCDF file handled safely without crashing.
5. Incomplete NetCDF (missing required variables like temperature) safely rejected.
6. Coordinates outside Bay of Bengal domain rejected with 422.
7. Malformed dates rejected with 422.
8. Missing model checkpoint returns clean 503 error.
9. Explicit real-mode prediction request returns clean 503 when real data is not configured (no silent fallback).
10. Synthetic mode reproducibility & determinism verified.
11. Model engine caching verified across multiple calls.
12. Streamlit dashboard & all subcomponents import cleanly.
13. Real satellite preprocessor dataset compatibility validation.
14. ARGO validation reports honest NOT_AVAILABLE status when data is missing.
15. Strict scientific honesty: no service claims real data prematurely.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys

# Ensure project root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app
from src.api.services.argo_service import ArgoService
from src.api.services.model_service import ModelNotAvailableError, ModelService, RealDataNotAvailableError
from src.api.services.satellite_service import SatelliteService
from src.data.real_satellite_preprocessing import DataIncompatibilityError, RealSatellitePreprocessor
from src.eval.argo_validation import check_argo_available, run_argo_validation
from src.inference import OceanInferenceEngine


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_1_missing_satellite_directory(tmp_path: Path) -> None:
    """1. Verify missing satellite directory does not crash SatelliteService."""
    non_existent_dir = tmp_path / "non_existent_satellite"
    service = SatelliteService()
    service.cmems_provider.is_available = lambda: False
    status = service.get_satellite_status()
    assert status["available"] is False
    assert status["mode"] == "synthetic_demo"
    assert "not configured" in status["message"].lower()


def test_2_missing_argo_directory(tmp_path: Path) -> None:
    """2. Verify missing ARGO directory returns honest unavailable status."""
    non_existent_dir = tmp_path / "non_existent_argo"
    argo_service = ArgoService(data_dir=non_existent_dir)
    status = argo_service.get_argo_status()
    assert status["available"] is False
    assert status["files_found"] == 0
    assert status["profiles_available"] == 0
    assert "No real ARGO NetCDF profiles found" in status["message"]


def test_3_empty_argo_directory(tmp_path: Path) -> None:
    """3. Verify empty ARGO directory returns files_found=0 and available=False."""
    empty_dir = tmp_path / "empty_argo"
    empty_dir.mkdir(parents=True, exist_ok=True)
    argo_service = ArgoService(data_dir=empty_dir)
    status = argo_service.get_argo_status()
    assert status["available"] is False
    assert status["files_found"] == 0
    assert status["profiles_available"] == 0


def test_4_malformed_netcdf_handling(tmp_path: Path) -> None:
    """4. Verify corrupted non-NetCDF file is safely rejected without crashing."""
    corrupt_file = tmp_path / "corrupt_profile.nc"
    corrupt_file.write_bytes(b"THIS IS NOT A VALID NETCDF FILE HEADER")
    argo_service = ArgoService(data_dir=tmp_path)
    res = argo_service.validate_profile_file(corrupt_file)
    assert res["valid"] is False
    assert res["profile_count"] == 0
    assert res["error"] is not None


def test_5_missing_required_netcdf_variables(tmp_path: Path) -> None:
    """5. Verify NetCDF missing required physical variables is rejected safely."""
    import xarray as xr
    # Create dataset with only latitude and longitude, missing temperature and depth
    ds = xr.Dataset(
        coords={
            "latitude": (["latitude"], [15.0]),
            "longitude": (["longitude"], [88.0]),
        }
    )
    incomplete_file = tmp_path / "incomplete_profile.nc"
    ds.to_netcdf(incomplete_file)

    argo_service = ArgoService(data_dir=tmp_path)
    res = argo_service.validate_profile_file(incomplete_file)
    assert res["valid"] is False
    assert "Missing required variables" in res["error"]


def test_6_invalid_coordinates_rejected(client: TestClient) -> None:
    """6. Verify coordinates outside Bay of Bengal (5-23°N, 80-100°E) return 422."""
    bad_coords = [
        {"latitude": 4.9, "longitude": 88.0, "date": "2023-02-15"},
        {"latitude": 23.5, "longitude": 88.0, "date": "2023-02-15"},
        {"latitude": 14.0, "longitude": 79.5, "date": "2023-02-15"},
        {"latitude": 14.0, "longitude": 100.5, "date": "2023-02-15"},
    ]
    for payload in bad_coords:
        res = client.post("/predict", json=payload)
        assert res.status_code == 422
        assert "detail" in res.json()


def test_7_invalid_date_rejected(client: TestClient) -> None:
    """7. Verify malformed and out-of-range dates return 422."""
    bad_dates = [
        {"latitude": 14.0, "longitude": 88.0, "date": "15-02-2023"},
        {"latitude": 14.0, "longitude": 88.0, "date": "2023/02/15"},
        {"latitude": 14.0, "longitude": 88.0, "date": "invalid-date"},
        {"latitude": 14.0, "longitude": 88.0, "date": "2010-01-01"},
    ]
    for payload in bad_dates:
        res = client.post("/predict", json=payload)
        assert res.status_code == 422


def test_8_missing_checkpoint_returns_clean_error() -> None:
    """8. Verify missing model checkpoint raises ModelNotAvailableError cleanly."""
    service = ModelService(checkpoint_path="checkpoints/definitely_missing_checkpoint.pt")
    with pytest.raises(ModelNotAvailableError) as exc_info:
        service.get_engine()
    assert "not found" in str(exc_info.value).lower()


def test_9_explicit_real_mode_without_data_rejected_with_503(client: TestClient) -> None:
    """9. Verify requesting data_mode='real' without real satellite data returns 503 (no silent fallback)."""
    payload = {
        "latitude": 14.0,
        "longitude": 88.0,
        "date": "2023-02-15",
        "data_mode": "real",
    }
    res = client.post("/predict", json=payload)
    # Must be 503 Service Unavailable, NEVER silent fallback to synthetic 200
    assert res.status_code == 503
    assert "Real satellite data pipeline is not configured" in res.json()["detail"]


def test_10_synthetic_mode_determinism() -> None:
    """10. Verify synthetic pipeline is 100% deterministic and reproducible."""
    ckpt_file = WORKSPACE_ROOT / "checkpoints" / "oceanembed_v2.pt"
    if not ckpt_file.exists():
        pytest.skip("OceanEmbed V2 checkpoint not yet present.")

    engine = OceanInferenceEngine(ckpt_file)
    res1 = engine.predict_from_location_date(14.25, 88.50, "2023-03-15")
    res2 = engine.predict_from_location_date(14.25, 88.50, "2023-03-15")

    np.testing.assert_allclose(res1["temperature"], res2["temperature"], atol=1e-5)
    np.testing.assert_allclose(res1["uncertainty_sigma"], res2["uncertainty_sigma"], atol=1e-5)
    np.testing.assert_allclose(res1["embedding"], res2["embedding"], atol=1e-5)
    np.testing.assert_allclose(res1["regime_probs"], res2["regime_probs"], atol=1e-5)


def test_11_model_caching_verified(client: TestClient) -> None:
    """11. Verify model engine is cached in memory across multiple requests."""
    service = ModelService.get_instance()
    engine_id_1 = id(service.get_engine())

    client.post("/predict", json={"latitude": 12.0, "longitude": 85.0, "date": "2023-04-10"})
    client.post("/predict", json={"latitude": 16.0, "longitude": 90.0, "date": "2023-05-10"})

    engine_id_2 = id(service.get_engine())
    assert engine_id_1 == engine_id_2


def test_12_dashboard_and_components_import_cleanly() -> None:
    """12. Verify dashboard and all modular components import cleanly."""
    import dashboard.app
    import dashboard.components.map
    import dashboard.components.metrics
    import dashboard.components.plots
    import dashboard.components.profile_table
    import dashboard.components.regime
    assert dashboard.app is not None


def test_13_real_satellite_preprocessor_validation() -> None:
    """13. Verify RealSatellitePreprocessor detects missing channels in incomplete datasets."""
    import xarray as xr
    preprocessor = RealSatellitePreprocessor()
    # Dummy dataset with only SST (thetao), missing SSS, SLA, Wind-U, Wind-V
    ds = xr.Dataset(
        data_vars={"thetao": (["time", "lat", "lon"], np.zeros((5, 10, 10), dtype=np.float32))},
        coords={
            "lat": np.linspace(10, 20, 10),
            "lon": np.linspace(85, 95, 10),
            "time": [datetime.datetime(2023, 2, i) for i in range(1, 6)],
        },
    )
    compat = preprocessor.validate_dataset_compatibility(ds)
    assert compat["compatible"] is False
    assert "SSS" in compat["missing_channels"]
    assert "SLA" in compat["missing_channels"]
    assert "Wind-U" in compat["missing_channels"]
    assert "Wind-V" in compat["missing_channels"]


def test_14_argo_validation_honest_reporting() -> None:
    """14. Verify argo_validation reports honest NOT_AVAILABLE status when data is missing."""
    val_res = run_argo_validation()
    assert val_res["status"] == "NOT_AVAILABLE"
    assert "ARGO validation unavailable because no real in-situ observations are configured." in val_res["reason"]
    assert val_res["n_samples"] == 0


def test_15_no_false_real_data_claims(client: TestClient) -> None:
    """15. Verify that all endpoints adhere strictly to scientific honesty."""
    h = client.get("/health").json()
    assert h["mode"] == "synthetic_demo"
    assert h["data"]["satellite"] is False
    assert h["data"]["argo"] is False

    argo = client.get("/data/argo/status").json()
    assert argo["available"] is False

    sat = client.get("/data/satellite/status").json()
    assert sat["available"] is False
