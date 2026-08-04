"""
Layer 6: Real-Time Inference Microservice.

Exposes a FastAPI application with a single ``/predict`` endpoint that:

    1. Accepts a JSON payload validated by the ``TransactionPayload`` Pydantic
       model (30 raw feature fields: V1–V28, Amount, Time).
    2. Applies the same feature transformations used during training
       (cyclical time encoding, hour-of-day bucketing, robust Amount scaling,
       log-Amount, and round-Amount flag) by loading the fitted transformer
       parameters from the model artefact directory.
    3. Calls the loaded XGBoost booster to produce a fraud probability.
    4. Compares the probability against the stored production threshold and
       returns a ``PredictionResponse`` with the fraud flag, raw probability,
       and applied threshold.

Design decisions:
    - The booster and configuration are loaded once at application startup
      via a FastAPI ``lifespan`` context manager so every request hits
      in-memory objects with no disk I/O on the hot path.
    - The production threshold is read from an artefact file
      (``artifacts/threshold.json``) written by the pipeline's Layer 5 step.
      If the file is absent, the default threshold of 0.5 is used and a
      warning is logged.
    - All field-level validation is delegated to Pydantic; the endpoint
      itself contains no manual validation code.
    - The application is run via ``uvicorn`` when this module is executed
      directly.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import PipelineConfig
from feature_engineering import FraudFeatureTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level mutable state populated by the lifespan handler
# ---------------------------------------------------------------------------

_booster: xgb.Booster | None = None
_transformer: FraudFeatureTransformer | None = None
_config: PipelineConfig | None = None
_production_threshold: float = 0.5

_THRESHOLD_ARTEFACT_FILENAME: str = "threshold.json"


# ---------------------------------------------------------------------------
# Pydantic I/O models
# ---------------------------------------------------------------------------


class TransactionPayload(BaseModel):
    """Validated input payload for a single credit card transaction.

    Fields mirror the raw Kaggle European Credit Card dataset columns.
    ``Time`` is in seconds elapsed since the first transaction in the dataset.
    ``Amount`` is the transaction value in EUR.  V1–V28 are PCA-transformed
    principal components provided by the dataset.

    All fields are required; no defaults are permitted so that the caller
    cannot inadvertently omit transaction data.

    Attributes:
        Time: Seconds elapsed since first transaction (≥ 0).
        Amount: Transaction amount in EUR (≥ 0).
        V1 … V28: PCA principal components (unbounded floats).
    """

    Time: float = Field(..., ge=0.0, description="Seconds since first transaction.")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in EUR.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    """Structured response returned by the ``/predict`` endpoint.

    Attributes:
        is_fraud: ``True`` when the predicted probability exceeds the
            production threshold.
        fraud_probability: Raw model output probability in [0, 1].
        threshold_applied: Production threshold used to derive ``is_fraud``.
    """

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    threshold_applied: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Structured response returned by the ``/health`` endpoint.

    Attributes:
        status: Always ``"ok"`` when the service is up and model is loaded.
        model_loaded: ``True`` when the booster has been loaded into memory.
        threshold: Active production threshold.
    """

    status: str
    model_loaded: bool
    threshold: float


# ---------------------------------------------------------------------------
# Lifespan handler — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the XGBoost booster, transformer, and threshold at startup.

    Uses the FastAPI ``lifespan`` interface (preferred over deprecated
    ``on_event`` decorators).  The booster is loaded from
    ``config.MODEL_SAVE_PATH``; transformer statistics are re-derived from the
    same config.  The production threshold is read from
    ``artifacts/threshold.json`` if present.

    Raises:
        RuntimeError: If the booster artefact does not exist or cannot be
            loaded, preventing the service from starting.
    """
    global _booster, _transformer, _config, _production_threshold

    try:
        _config = PipelineConfig()
        logger.info("API startup: PipelineConfig loaded.")

        model_path: Path = _config.MODEL_SAVE_PATH
        if not model_path.exists():
            raise RuntimeError(
                f"Model artefact not found at {model_path}. "
                "Run the training pipeline before starting the API."
            )

        loaded_booster: xgb.Booster = xgb.Booster()
        loaded_booster.load_model(str(model_path))
        _booster = loaded_booster
        logger.info("Booster loaded from: %s", model_path)

        stats_path: Path = _config.TRANSFORMER_STATS_PATH
        if not stats_path.exists():
            raise RuntimeError(
                f"Transformer stats artefact not found at {stats_path}. "
                "Run the full training pipeline first to generate this file."
            )
        _transformer = FraudFeatureTransformer(_config)
        _transformer.load_state(stats_path)
        logger.info(
            "Transformer state loaded. amount_median=%.4f amount_iqr=%.4f",
            _transformer.amount_median_,
            _transformer.amount_iqr_,
        )

        threshold_path: Path = (
            _config.MODEL_SAVE_PATH.parent / _THRESHOLD_ARTEFACT_FILENAME
        )
        if threshold_path.exists():
            try:
                with open(threshold_path, "r", encoding="utf-8") as fh:
                    threshold_data: dict[str, Any] = json.load(fh)
                raw_threshold: Any = threshold_data.get("optimal_threshold", 0.5)
                _production_threshold = float(raw_threshold)
                logger.info(
                    "Production threshold loaded from artefact: %.4f",
                    _production_threshold,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to parse threshold artefact (%s). "
                    "Falling back to default threshold 0.5.",
                    exc,
                )
                _production_threshold = 0.5
        else:
            logger.warning(
                "Threshold artefact not found at %s. "
                "Using default threshold 0.5. Run the full pipeline to "
                "generate a cost-optimal threshold.",
                threshold_path,
            )
            _production_threshold = 0.5

        logger.info("API startup complete. Threshold: %.4f", _production_threshold)

    except RuntimeError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during API startup.")
        raise RuntimeError(f"API lifespan startup failed: {exc}") from exc

    yield  # application runs here

    logger.info("API shutdown: releasing resources.")
    _booster = None
    _transformer = None
    _config = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app: FastAPI = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time inference microservice for the XGBoost fraud detection "
        "pipeline.  Submit a single transaction's raw features and receive a "
        "binary fraud decision alongside the raw probability score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Return the liveness status of the inference service.

    Confirms whether the booster has been loaded into memory and which
    production threshold is active.  Intended for use by load-balancer
    health probes and monitoring dashboards.

    Returns:
        ``HealthResponse`` with ``status="ok"``, ``model_loaded`` flag, and
        the active ``threshold``.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_booster is not None,
        threshold=_production_threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: TransactionPayload) -> PredictionResponse:
    """Score a single credit card transaction for fraud.

    Workflow:
        1. Validate the incoming JSON payload via ``TransactionPayload``
           (handled automatically by FastAPI / Pydantic before this function
           is called).
        2. Construct a single-row ``pd.DataFrame`` from the payload using the
           expected column order.
        3. Apply in-memory feature transformations (using fitted transformer
           stats loaded from ``transformer_stats.json`` at startup):
               - Cyclical time encoding: ``Time`` → ``Time_sin``, ``Time_cos``
               - Hour bucket: ``Time`` → ``Time_hour`` (0–23)
               - Amount engineering: ``Amount`` → ``Amount_scaled``,
                 ``Amount_log1p``, ``Amount_is_round``
        4. Pass the transformed row to the booster to obtain a probability.
        5. Apply the production threshold and return the decision.

    Args:
        payload: Validated ``TransactionPayload`` parsed from the request body.

    Returns:
        ``PredictionResponse`` containing ``is_fraud``, ``fraud_probability``,
        and ``threshold_applied``.

    Raises:
        HTTPException 503: If the booster or transformer is not loaded (service
            not yet ready).
        HTTPException 500: If any transformation or inference step fails
            unexpectedly.
    """
    if _booster is None or _transformer is None or _config is None:
        logger.error("Predict called before booster/transformer is loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The service is not ready.",
        )

    try:
        raw_data: dict[str, float] = payload.model_dump()
        feature_order: list[str] = (
            ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        )
        row_df: pd.DataFrame = pd.DataFrame([raw_data])[feature_order]

        for col in row_df.select_dtypes(include=["float64"]).columns:
            row_df[col] = row_df[col].astype("float32")

        transformed_df: pd.DataFrame = _transformer.transform(row_df)

        dmat: xgb.DMatrix = xgb.DMatrix(
            transformed_df.to_numpy(dtype=np.float32),
            feature_names=transformed_df.columns.tolist(),
        )
        fraud_proba: float = float(_booster.predict(dmat)[0])

        is_fraud: bool = fraud_proba >= _production_threshold

        logger.info(
            "Prediction complete. probability=%.6f threshold=%.4f is_fraud=%s",
            fraud_proba,
            _production_threshold,
            is_fraud,
        )

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_proba,
            threshold_applied=_production_threshold,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------


def save_threshold_artefact(optimal_threshold: float, config: PipelineConfig) -> None:
    """Persist the optimal threshold to a JSON file in the artefacts directory.

    Called by the pipeline orchestrator after ``CostMatrixEvaluator`` has
    determined the production threshold so the API can load it at startup.

    Args:
        optimal_threshold: Float in (0, 1) representing the cost-minimising
            decision boundary.
        config: ``PipelineConfig`` supplying the artefact directory path.
    """
    threshold_path: Path = config.MODEL_SAVE_PATH.parent / _THRESHOLD_ARTEFACT_FILENAME
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, float] = {"optimal_threshold": optimal_threshold}
    with open(threshold_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(
        "Threshold artefact saved: %.4f → %s", optimal_threshold, threshold_path
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_serving:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
