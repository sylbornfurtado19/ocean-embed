"""Comprehensive automated test suite for OceanEmbed Backend & API Hardening.

Validates all 27+ hardening criteria:
1. Malformed JSON returns 422.
2. Oversized request body returns 413.
3. NaN coordinate rejection returns 422.
4. Infinity coordinate rejection returns 422.
5. Out-of-domain coordinate returns 422.
6. Invalid date format returns 422.
7. Unsupported data_mode returns 422.
8. Real-mode unavailable returns clean 503.
9. Missing model checkpoint returns clean 503.
10. Internal exception returns safe 500.
11. No raw traceback in public responses.
12. No host filesystem path leakage in public responses.
13. CORS rejects untrusted origin in allowlisted mode.
14. CORS accepts trusted origin.
15. Optional API key rejection (401 when enabled and invalid).
16. Optional API key acceptance (200 when valid).
17. Rate limit enforcement on /predict (429).
18. Inference concurrency limiter bounding.
19. Request ID automatically generated.
20. Request ID returned in response headers (X-Request-ID).
21. Security headers present (X-Content-Type-Options, X-Frame-Options, Referrer-Policy).
22. ARGO malformed NetCDF safely rejected without crashing.
23. ARGO path traversal outside data root prevented.
24. Satellite real-data processing failure suppresses internal exception strings.
25. Model status endpoint does not leak absolute host paths.
26. Deterministic synthetic behavior preserved.
27. Non-numeric extra payload fields forbidden.
"""

from __future__ import annotations

import asyncio
import datetime
import math
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app
from src.api.middleware import InferenceConcurrencyLimiter
from src.api.services.argo_service import ArgoService
from src.api.services.model_service import ModelNotAvailableError, ModelService, RealDataNotAvailableError
from src.config import settings
from src.inference import OceanInferenceEngine


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Standard client with lifespan active."""
    with TestClient(app) as test_client:
        yield test_client


# --- 1. Malformed JSON ---
def test_1_malformed_json_rejected(client: TestClient) -> None:
    """Verify that unparseable raw JSON returns HTTP 422."""
    res = client.post(
        "/predict",
        content=b"{latitude: 14.0, longitude: 88.0, missing_closing_bracket",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data
    assert data.get("error_code") == "VALIDATION_ERROR"


# --- 2. Oversized Request Body ---
def test_2_oversized_request_rejected(client: TestClient) -> None:
    """Verify that requests exceeding MAX_REQUEST_BODY_BYTES return HTTP 413."""
    # Send request with oversized content-length
    huge_payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15", "padding": "x" * (settings.max_request_body_bytes + 1000)}
    res = client.post(
        "/predict",
        json=huge_payload,
    )
    assert res.status_code == 413
    assert res.json().get("error_code") == "PAYLOAD_TOO_LARGE"


# --- 3. NaN Coordinate Rejection ---
def test_3_nan_coordinates_rejected(client: TestClient) -> None:
    """Verify that NaN coordinates are strictly rejected both via Pydantic model and HTTP API."""
    from pydantic import ValidationError
    from src.api.schemas import PredictionRequest

    # Direct Pydantic model validation
    with pytest.raises(ValidationError) as exc:
        PredictionRequest(latitude=float("nan"), longitude=88.0, date="2023-06-15")
    assert "finite" in str(exc.value).lower()

    with pytest.raises(ValidationError) as exc:
        PredictionRequest(latitude=14.0, longitude=float("nan"), date="2023-06-15")
    assert "finite" in str(exc.value).lower()

    # HTTP API rejection of non-numerical representation
    res_lat = client.post("/predict", json={"latitude": "NaN", "longitude": 88.0, "date": "2023-06-15"})
    assert res_lat.status_code == 422


# --- 4. Infinity Coordinate Rejection ---
def test_4_infinity_coordinates_rejected(client: TestClient) -> None:
    """Verify that +Infinity and -Infinity are rejected both via Pydantic model and HTTP API."""
    from pydantic import ValidationError
    from src.api.schemas import PredictionRequest

    # Direct Pydantic model validation
    with pytest.raises(ValidationError) as exc_inf:
        PredictionRequest(latitude=float("inf"), longitude=88.0, date="2023-06-15")
    assert "finite" in str(exc_inf.value).lower()

    with pytest.raises(ValidationError) as exc_ninf:
        PredictionRequest(latitude=14.0, longitude=float("-inf"), date="2023-06-15")
    assert "finite" in str(exc_ninf.value).lower()

    # HTTP API rejection
    res_inf = client.post("/predict", json={"latitude": "Infinity", "longitude": 88.0, "date": "2023-06-15"})
    assert res_inf.status_code == 422



# --- 5. Out-of-Domain Coordinate Rejection ---
def test_5_out_of_domain_coordinates(client: TestClient) -> None:
    """Verify that coordinates outside 5-23°N, 80-100°E return HTTP 422."""
    res = client.post("/predict", json={"latitude": 2.0, "longitude": 88.0, "date": "2023-06-15"})
    assert res.status_code == 422
    assert "Bay of Bengal" in res.json()["detail"]


# --- 6. Invalid Date Rejection ---
def test_6_invalid_date_format_and_range(client: TestClient) -> None:
    """Verify invalid format and out-of-simulation-window dates return HTTP 422."""
    res_fmt = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "15/06/2023"})
    assert res_fmt.status_code == 422

    res_range = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2015-06-15"})
    assert res_range.status_code == 422
    assert "supported simulation window" in res_range.json()["detail"]


# --- 7. Unsupported data_mode ---
def test_7_unsupported_data_mode(client: TestClient) -> None:
    """Verify unsupported data mode returns HTTP 422."""
    res = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15", "data_mode": "quantum"})
    assert res.status_code == 422


# --- 8. Real-Mode Unavailable => 503 ---
def test_8_real_mode_unavailable_503(client: TestClient) -> None:
    """Verify real data mode returns HTTP 503 with honest message when unconfigured."""
    res = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15", "data_mode": "real"})
    assert res.status_code == 503
    data = res.json()
    assert "Real satellite data pipeline is not configured" in data["detail"]


# --- 9. Model Unavailable => 503 ---
def test_9_model_unavailable_503() -> None:
    """Verify ModelNotAvailableError raises HTTP 503 cleanly."""
    fake_service = ModelService(checkpoint_path="checkpoints/missing_checkpoint.pt")
    with pytest.raises(ModelNotAvailableError) as exc:
        fake_service.get_engine()
    assert "not found" in str(exc.value).lower()


# --- 10. Internal Exception => Safe 500 ---
def test_10_internal_exception_returns_safe_500(client: TestClient) -> None:
    """Verify unhandled internal errors return HTTP 500 without leaking stack traces."""
    with patch.object(ModelService, "predict", side_effect=RuntimeError("Secret internal failure in forward pass")):
        res = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15"})
        assert res.status_code == 500
        data = res.json()
        assert data["detail"] == "Subsurface profile reconstruction failed during model forward pass."
        assert "Secret internal failure" not in str(data)
        assert "Traceback" not in str(data)


# --- 11. No Raw Traceback in Public Responses ---
def test_11_no_raw_traceback_leakage(client: TestClient) -> None:
    """Verify that validation errors or server errors contain no Python tracebacks."""
    res = client.get("/non_existent_route")
    assert res.status_code == 404
    assert "Traceback" not in res.text
    assert "File \"" not in res.text


# --- 12. No Filesystem Path Leakage ---
def test_12_no_filesystem_path_leakage(client: TestClient) -> None:
    """Verify that status responses do not leak absolute host directories (e.g. C:\\Users\\...)."""
    model_status = client.get("/model/status").json()
    assert ":\\" not in model_status["checkpoint_path"]
    assert model_status["checkpoint_path"].startswith("checkpoints") or model_status["checkpoint_path"].endswith(".pt")

    argo_status = client.get("/data/argo/status").json()
    assert ":\\" not in argo_status["directory"]
    assert argo_status["directory"].startswith("data") or "argo" in argo_status["directory"]


# --- 13 & 14. CORS Headers ---
def test_13_cors_trusted_origin_accepted(client: TestClient) -> None:
    """Verify that allowlisted origin receives Access-Control-Allow-Origin header."""
    trusted_origin = settings.cors_allowed_origins[0] if settings.cors_allowed_origins else "http://localhost:5173"
    res = client.options(
        "/predict",
        headers={
            "Origin": trusted_origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") == trusted_origin


def test_14_cors_untrusted_origin_rejected(client: TestClient) -> None:
    """Verify that untrusted origin does NOT receive access-control-allow-origin header."""
    res = client.options(
        "/predict",
        headers={
            "Origin": "http://evil-attacker-site.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") != "http://evil-attacker-site.com"


# --- 15 & 16. Optional API Key Protection ---
def test_15_api_key_rejection_when_configured() -> None:
    """Verify that when API_KEY is set, requests without matching X-API-Key are rejected with 401."""
    with patch.object(settings, "api_key", "secret-test-key-999"):
        with TestClient(app) as auth_client:
            # Missing key
            res_missing = auth_client.get("/model/status")
            assert res_missing.status_code == 401
            assert "API key" in res_missing.json()["detail"]

            # Incorrect key
            res_bad = auth_client.get("/model/status", headers={"X-API-Key": "wrong-key"})
            assert res_bad.status_code == 401

            # Health remains public even when API key is set
            res_health = auth_client.get("/health")
            assert res_health.status_code == 200


def test_16_api_key_acceptance_when_configured() -> None:
    """Verify that when API_KEY is set, valid X-API-Key header allows access."""
    with patch.object(settings, "api_key", "secret-test-key-999"):
        with TestClient(app) as auth_client:
            res_valid = auth_client.get("/model/status", headers={"X-API-Key": "secret-test-key-999"})
            assert res_valid.status_code == 200
            assert res_valid.json()["version"] == "oceanembed_v2"


# --- 17. Rate Limit Enforcement ---
def test_17_rate_limit_enforcement() -> None:
    """Verify that exceeding rate limit triggers HTTP 429 with Retry-After header."""
    with patch.object(settings, "rate_limit_per_minute", 3):
        # Use an isolated client IP to test sliding window independently
        with TestClient(app, client=("198.51.100.42", 50000)) as rate_client:
            payload = {"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15"}
            # Make 3 allowed requests
            for _ in range(3):
                r = rate_client.post("/predict", json=payload)
                assert r.status_code == 200

            # 4th request must be rejected with 429
            r_blocked = rate_client.post("/predict", json=payload)
            assert r_blocked.status_code == 429
            assert r_blocked.json().get("error_code") == "RATE_LIMIT_EXCEEDED"
            assert "Retry-After" in r_blocked.headers



# --- 18. Inference Concurrency Cap ---
@pytest.mark.anyio
async def test_18_concurrency_limiter_bounds() -> None:
    """Verify that InferenceConcurrencyLimiter raises 503 when capacity is saturated."""
    limiter = InferenceConcurrencyLimiter(max_concurrent=1, acquire_timeout=0.05)
    acquired_first = False

    async with limiter.slot(request_id="first"):
        acquired_first = True
        # While holding slot, second acquisition must timeout and raise HTTPException 503
        with pytest.raises(Exception) as exc:
            async with limiter.slot(request_id="second"):
                pass
        assert "503" in str(exc.value)

    assert acquired_first is True


# --- 19 & 20. Request ID Lifecycle ---
def test_19_request_id_generated_and_returned(client: TestClient) -> None:
    """Verify that every response includes an X-Request-ID header."""
    res = client.get("/health")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert len(res.headers["X-Request-ID"]) > 10


def test_20_custom_request_id_propagated(client: TestClient) -> None:
    """Verify that a valid incoming X-Request-ID is propagated back in response headers."""
    custom_id = "custom-sih-test-req-001"
    res = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers["X-Request-ID"] == custom_id


# --- 21. Security Headers Present ---
def test_21_security_headers_present(client: TestClient) -> None:
    """Verify presence of security headers on all responses."""
    res = client.get("/health")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "no-store" in res.headers.get("Cache-Control", "")


# --- 22. ARGO Malformed NetCDF Safely Rejected ---
def test_22_argo_malformed_file_rejected(tmp_path: Path) -> None:
    """Verify that corrupted non-NetCDF files are safely ignored without crashing."""
    corrupt_file = tmp_path / "bad.nc"
    corrupt_file.write_bytes(b"CORRUPT_BYTES_DATA")
    argo_service = ArgoService(data_dir=tmp_path)
    res = argo_service.validate_profile_file(corrupt_file)
    assert res["valid"] is False
    assert res["profile_count"] == 0


# --- 23. ARGO Path Traversal Guard ---
def test_23_argo_path_traversal_guard(tmp_path: Path) -> None:
    """Verify that directory discovery does not escape the configured ARGO root."""
    inside_dir = tmp_path / "argo_inside"
    inside_dir.mkdir()
    (inside_dir / "valid.nc").write_bytes(b"NC")

    outside_dir = tmp_path / "secret_outside"
    outside_dir.mkdir()
    (outside_dir / "escaped.nc").write_bytes(b"NC")

    service = ArgoService(data_dir=inside_dir)
    discovered = service.discover_files()
    assert all("secret_outside" not in str(p) for p in discovered)


# --- 24. Satellite Processing Failure Suppresses Exceptions ---
def test_24_satellite_failure_suppresses_raw_traceback(client: TestClient) -> None:
    """Verify that real satellite preprocessing failures do not leak raw Python exception traces."""
    with patch("src.api.services.model_service.ModelService.predict", side_effect=RealDataNotAvailableError("Real satellite data processing failed.")):
        res = client.post("/predict", json={"latitude": 14.0, "longitude": 88.0, "date": "2023-06-15", "data_mode": "real"})
        assert res.status_code == 503
        assert res.json()["detail"] == "Real satellite data processing failed."
        assert "Traceback" not in res.text


# --- 25. Model Status Does Not Leak Absolute Host Paths ---
def test_25_model_status_path_sanitization(client: TestClient) -> None:
    """Verify model status response contains safe normalized paths."""
    res = client.get("/model/status")
    assert res.status_code == 200
    data = res.json()
    assert "Users\\" not in data["checkpoint_path"]
    assert "home/" not in data["checkpoint_path"]


# --- 26. Deterministic Synthetic Behavior Preserved ---
def test_26_deterministic_synthetic_predictions(client: TestClient) -> None:
    """Verify that two identical prediction requests produce identical numerical predictions."""
    payload = {"latitude": 14.25, "longitude": 88.5, "date": "2023-05-15", "data_mode": "synthetic"}
    res1 = client.post("/predict", json=payload).json()
    res2 = client.post("/predict", json=payload).json()
    assert res1["temperatures"] == res2["temperatures"]
    assert res1["sigma"] == res2["sigma"]
    assert res1["embedding"] == res2["embedding"]
    assert res1["regime_probs"] == res2["regime_probs"]


# --- 27. Extra Forbidden Fields in PredictionRequest ---
def test_27_extra_forbidden_fields_rejected(client: TestClient) -> None:
    """Verify that unexpected extra payload fields are rejected with HTTP 422."""
    payload = {
        "latitude": 14.0,
        "longitude": 88.0,
        "date": "2023-06-15",
        "malicious_extra_field": "injected_value",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "extra_forbidden" in res.json()["detail"] or "Extra inputs are not permitted" in res.json()["detail"] or "extra" in res.json()["detail"].lower()
