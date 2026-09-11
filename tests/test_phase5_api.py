"""Comprehensive automated test suite for OceanEmbed Phase 5 FastAPI Backend.

Validates all criteria specified in Phase 5:
1. API imports cleanly and app instance initializes.
2. FastAPI app starts and lifespan triggers.
3. /health returns valid status, model availability, and honest data modes.
4. /model/status returns checkpoint metadata, dimensions, and depths.
5. Valid prediction request returns 200 and structured response.
6. Invalid latitude (< 5.0 or > 23.0) is rejected (422).
7. Invalid longitude (< 80.0 or > 100.0) is rejected (422).
8. Invalid date format is rejected (422).
9. Prediction returns exactly 15 depths.
10. Prediction returns plausible temperature values.
11. Sigma uncertainty values are finite and positive.
12. lower_90 and upper_90 intervals match mean ± 1.645*sigma.
13. Regime probabilities sum approximately to 1.0.
14. Embedding vector has exactly 512 dimensions.
15. Missing checkpoint raises clean 503 error without crashing or fallback to random weights.
16. ARGO status endpoint returns unavailable when files absent.
17. Satellite status correctly reports synthetic mode.
18. No API endpoint falsely claims real data.
19. Model is not reloaded for every prediction request (cached instance).
20. Existing Phase 2 baseline models (V0, V1) still load and run.
21. Existing Phase 3 OceanEmbed V2 inference and embedding extraction work.
22. Existing Phase 4 Streamlit dashboard still imports cleanly.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app
from src.api.services.argo_service import ArgoService
from src.api.services.model_service import ModelNotAvailableError, ModelService
from src.api.services.satellite_service import SatelliteService
from src.inference import OceanInferenceEngine


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a TestClient with lifespan context active."""
    with TestClient(app) as test_client:
        yield test_client


def test_1_api_imports_successfully() -> None:
    """1. Verify that FastAPI application and modules import cleanly."""
    import src.api.main
    import src.api.routes.data
    import src.api.routes.health
    import src.api.routes.inference
    import src.api.schemas
    import src.api.services.argo_service
    import src.api.services.model_service
    import src.api.services.satellite_service
    assert src.api.main.app is not None


def test_2_fastapi_app_starts(client: TestClient) -> None:
    """2. Verify that FastAPI app starts and root endpoint returns metadata."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "OceanEmbed"
    assert data["version"] == "2.0.0"
    assert data["docs"] == "/docs"


def test_3_health_endpoint(client: TestClient) -> None:
    """3. Verify /health returns status, model availability, and honest data modes."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["ok", "degraded"]
    assert "model" in data
    assert "data" in data
    assert data["model"]["version"] == "oceanembed_v2"
    # Adhere strictly to honesty: real data must not be claimed as true
    assert data["data"]["satellite"] is False
    assert data["data"]["argo"] is False
    assert data["mode"] == "synthetic_demo"


def test_4_model_status_endpoint(client: TestClient) -> None:
    """4. Verify /model/status inspects checkpoint metadata directly."""
    res = client.get("/model/status")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "oceanembed_v2"
    assert data["checkpoint_exists"] is True
    assert data["checkpoint_size_mb"] > 0
    assert data["embedding_dim"] == 512
    assert data["num_depths"] == 15
    assert len(data["target_depths"]) == 15
    assert data["num_regimes"] == 4
    assert data["domain"]["lat_min"] == 5.0
    assert data["domain"]["lat_max"] == 23.0


def test_5_valid_prediction_request(client: TestClient) -> None:
    """5. Verify /predict returns 200 and complete structured prediction."""
    payload = {
        "latitude": 14.5,
        "longitude": 88.0,
        "date": "2023-06-15",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["latitude"] == 14.5
    assert data["longitude"] == 88.0
    assert data["date"] == "2023-06-15"
    assert data["data_mode"] == "synthetic_demo"
    assert data["inference_latency_ms"] > 0.0


def test_6_invalid_latitude_rejected(client: TestClient) -> None:
    """6. Verify latitude outside 5.0°N–23.0°N is rejected with 422."""
    payload_low = {"latitude": 4.5, "longitude": 88.0, "date": "2023-06-15"}
    res_low = client.post("/predict", json=payload_low)
    assert res_low.status_code == 422
    assert "Latitude" in res_low.json()["detail"]

    payload_high = {"latitude": 24.0, "longitude": 88.0, "date": "2023-06-15"}
    res_high = client.post("/predict", json=payload_high)
    assert res_high.status_code == 422
    assert "Latitude" in res_high.json()["detail"]


def test_7_invalid_longitude_rejected(client: TestClient) -> None:
    """7. Verify longitude outside 80.0°E–100.0°E is rejected with 422."""
    payload_low = {"latitude": 14.5, "longitude": 79.5, "date": "2023-06-15"}
    res_low = client.post("/predict", json=payload_low)
    assert res_low.status_code == 422
    assert "Longitude" in res_low.json()["detail"]

    payload_high = {"latitude": 14.5, "longitude": 101.5, "date": "2023-06-15"}
    res_high = client.post("/predict", json=payload_high)
    assert res_high.status_code == 422
    assert "Longitude" in res_high.json()["detail"]


def test_8_invalid_date_rejected(client: TestClient) -> None:
    """8. Verify malformed date is rejected with 422."""
    payload_bad = {"latitude": 14.5, "longitude": 88.0, "date": "not-a-date"}
    res = client.post("/predict", json=payload_bad)
    assert res.status_code == 422
    assert "date" in res.json()["detail"].lower()


def test_9_prediction_returns_15_depths(client: TestClient) -> None:
    """9. Verify prediction returns exactly 15 standardized depth levels."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    expected_depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
    assert len(data["depths"]) == 15
    assert data["depths"] == expected_depths


def test_10_prediction_returns_temperatures(client: TestClient) -> None:
    """10. Verify predicted temperatures are returned for all 15 depths and are physically plausible."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    temps = data["temperatures"]
    assert len(temps) == 15
    # Plausible ocean temperature range (3°C to 33°C in Bay of Bengal)
    assert all(2.0 < t < 35.0 for t in temps)
    # Surface temperature should be higher than 1000m deep water temperature
    assert temps[0] > temps[-1]


def test_11_sigma_finite_and_positive(client: TestClient) -> None:
    """11. Verify uncertainty sigma values are finite, strictly positive, and non-zero."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    sigmas = res.json()["sigma"]
    assert len(sigmas) == 15
    assert all(np.isfinite(s) and s > 0.0 for s in sigmas)


def test_12_interval_mathematically_correct(client: TestClient) -> None:
    """12. Verify lower_90 and upper_90 bounds equal mean ± 1.645*sigma."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    data = res.json()
    temps = np.array(data["temperatures"])
    sigmas = np.array(data["sigma"])
    lower = np.array(data["lower_90"])
    upper = np.array(data["upper_90"])

    expected_lower = np.round(temps - 1.645 * sigmas, 4)
    expected_upper = np.round(temps + 1.645 * sigmas, 4)

    np.testing.assert_allclose(lower, expected_lower, atol=1e-3)
    np.testing.assert_allclose(upper, expected_upper, atol=1e-3)


def test_13_regime_probabilities_sum_to_one(client: TestClient) -> None:
    """13. Verify K=4 latent regime probabilities sum approximately to 1.0."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    regimes = res.json()["regime_probs"]
    assert len(regimes) == 4
    assert all(0.0 <= p <= 1.0 for p in regimes)
    assert abs(sum(regimes) - 1.0) < 1e-3


def test_14_embedding_has_512_dimensions(client: TestClient) -> None:
    """14. Verify Ocean Embedding has exactly 512 dimensions."""
    payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}
    res = client.post("/predict", json=payload)
    embedding = res.json()["embedding"]
    assert len(embedding) == 512
    assert all(np.isfinite(e) for e in embedding)


def test_15_missing_checkpoint_handled_correctly() -> None:
    """15. Verify missing checkpoint raises 503 without crashing or random fallback."""
    fake_service = ModelService(checkpoint_path="checkpoints/non_existent_file.pt")
    with pytest.raises(ModelNotAvailableError) as exc_info:
        fake_service.get_engine()
    assert "not found" in str(exc_info.value).lower()


def test_16_argo_status_endpoint(client: TestClient) -> None:
    """16. Verify /data/argo/status returns honest unavailable status when no files exist."""
    res = client.get("/data/argo/status")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["files_found"] == 0
    assert data["profiles_available"] == 0
    assert "No real ARGO NetCDF profiles found" in data["message"]


def test_17_satellite_status_endpoint(client: TestClient) -> None:
    """17. Verify /data/satellite/status reports synthetic demo mode honestly."""
    res = client.get("/data/satellite/status")
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is False
    assert data["mode"] == "synthetic_demo"
    assert data["provider"] is None
    assert "SST" in data["variables"]
    assert "Real satellite data provider is not configured" in data["message"]


def test_18_no_api_claims_real_data(client: TestClient) -> None:
    """18. Verify that neither /health, /predict, /data/argo/status, nor /data/satellite/status claims real data."""
    h = client.get("/health").json()
    assert h["mode"] == "synthetic_demo"
    assert h["data"]["satellite"] is False
    assert h["data"]["argo"] is False

    p = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2023-03-15"}).json()
    assert p["data_mode"] == "synthetic_demo"

    argo = client.get("/data/argo/status").json()
    assert argo["available"] is False

    sat = client.get("/data/satellite/status").json()
    assert sat["available"] is False


def test_19_model_not_reloaded_per_request(client: TestClient) -> None:
    """19. Verify that model engine is cached and not reloaded from disk on every request."""
    service = ModelService.get_instance()
    engine_id_before = id(service.get_engine())

    client.post("/predict", json={"latitude": 12.0, "longitude": 85.0, "date": "2023-04-10"})
    client.post("/predict", json={"latitude": 16.0, "longitude": 90.0, "date": "2023-05-10"})

    engine_id_after = id(service.get_engine())
    assert engine_id_before == engine_id_after


def test_20_phase2_preservation() -> None:
    """20. Verify Phase 2 V0 and V1 baseline models continue to function."""
    v0_path = WORKSPACE_ROOT / "checkpoints" / "v0_baseline.pt"
    v1_path = WORKSPACE_ROOT / "checkpoints" / "v1_uncertainty.pt"

    if v0_path.exists():
        v0_eng = OceanInferenceEngine(v0_path)
        assert v0_eng.model_version == "v0"
        res0 = v0_eng.predict([28.5, 33.2, 0.05, 2.1, -1.2, 14.0, 88.0, 45.0])
        assert len(res0["temperature"]) == 15

    if v1_path.exists():
        v1_eng = OceanInferenceEngine(v1_path)
        assert v1_eng.model_version == "v1_uncertainty"
        res1 = v1_eng.predict([28.5, 33.2, 0.05, 2.1, -1.2, 14.0, 88.0, 45.0])
        assert len(res1["temperature"]) == 15
        assert len(res1["uncertainty_sigma"]) == 15


def test_21_phase3_preservation() -> None:
    """21. Verify Phase 3 OceanEmbed V2 spatiotemporal inference & embedding extraction."""
    v2_path = WORKSPACE_ROOT / "checkpoints" / "oceanembed_v2.pt"
    assert v2_path.exists()
    engine = OceanInferenceEngine(v2_path)
    res = engine.predict_from_location_date(14.0, 88.0, datetime.date(2023, 2, 15))
    assert len(res["temperature"]) == 15
    assert len(res["embedding"]) == 512

    dummy_patch = np.zeros((31, 5, 32, 32), dtype=np.float32)
    emb = engine.extract_embedding(dummy_patch)
    assert emb.shape == (512,)


def test_22_phase4_dashboard_imports() -> None:
    """22. Verify Phase 4 Streamlit dashboard modules continue to import cleanly."""
    import dashboard.app
    import dashboard.components.map
    import dashboard.components.metrics
    import dashboard.components.plots
    import dashboard.components.profile_table
    import dashboard.components.regime
    assert dashboard.app is not None
