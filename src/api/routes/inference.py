import logging
import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, status
import numpy as np

from src.api.dependencies import get_model_service
from src.api.middleware import inference_concurrency_limiter
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.security import verify_api_key
from src.api.services.model_service import ModelNotAvailableError, ModelService, RealDataNotAvailableError

logger = logging.getLogger("oceanembed.api.inference")
router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Reconstruct subsurface ocean temperature profile",
    description=(
        "Executes OceanEmbed V2 neural inference on a 31-day spatiotemporal sequence to predict "
        "the 15-depth vertical temperature profile, 90% Gaussian predictive interval (±1.645σ), "
        "K=4 soft latent regime context probabilities, and continuous 512-D Ocean Embedding."
    ),
)
async def predict_subsurface_profile(
    request: PredictionRequest,
    raw_request: Request,
    _auth: str | None = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Run OceanEmbed V2 inference for requested coordinates and date with concurrency bounding."""
    req_id = getattr(raw_request.state, "request_id", "req-predict")
    try:
        async with inference_concurrency_limiter.slot(request_id=req_id):
            result = await anyio.to_thread.run_sync(
                model_service.predict,
                request.latitude,
                request.longitude,
                request.date,
                request.data_mode,
            )
    except ModelNotAvailableError as err:
        logger.error("[%s] Inference requested but model is unavailable: %s", req_id, err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OceanEmbed model unavailable: {err}",
        )
    except RealDataNotAvailableError as err:
        logger.warning("[%s] Real data inference requested but real data is unavailable: %s", req_id, err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except HTTPException:
        raise
    except Exception as ex:
        logger.error("[%s] Unexpected error during inference execution: %s", req_id, ex, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Subsurface profile reconstruction failed during model forward pass.",
        )


    temps = np.asarray(result["temperature"], dtype=float)
    sigmas = np.asarray(result["uncertainty_sigma"], dtype=float)

    lower_90 = np.round(temps - 1.645 * sigmas, 4).tolist()
    upper_90 = np.round(temps + 1.645 * sigmas, 4).tolist()

    return PredictionResponse(
        latitude=float(result["latitude"]),
        longitude=float(result["longitude"]),
        date=str(result["prediction_date"]),
        depths=[float(d) for d in result["depths"]],
        temperatures=[round(float(t), 4) for t in temps],
        sigma=[round(float(s), 4) for s in sigmas],
        lower_90=lower_90,
        upper_90=upper_90,
        regime_probs=[round(float(p), 4) for p in result["regime_probs"]],
        embedding=[round(float(e), 4) for e in result["embedding"]],
        climatology_prior=[round(float(c), 4) for c in result["climatology_prior"]] if "climatology_prior" in result else None,
        anomaly=[round(float(a), 4) for a in result["anomaly"]] if "anomaly" in result else None,
        surface_inputs=result.get("surface_inputs"),
        inference_latency_ms=float(result.get("inference_time_ms", 0.0)),
        data_mode=str(result.get("data_mode", "synthetic_demo")),
    )
