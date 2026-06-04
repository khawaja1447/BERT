"""
FastAPI inference server for BERT Sentiment Analysis Pipeline.

Endpoints:
  POST /predict          - Single text prediction
  POST /predict/batch    - Batch prediction (up to 64 texts)
  GET  /predict/stream   - SSE streaming (word-by-word analysis)
  GET  /health           - Health check
  GET  /metrics          - Latency & throughput stats
  GET  /docs             - Swagger UI (auto)

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    SentimentResult,
)
from src.utils import LatencyTracker, PredictionCounter, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model")

pipeline = None
latency_tracker = LatencyTracker()
counter = PredictionCounter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        from src.inference import SentimentPipeline
        use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"
        pipeline = SentimentPipeline.from_pretrained(MODEL_PATH, use_onnx=use_onnx)
        logger.info("Sentiment pipeline ready")
    except Exception as e:
        logger.warning(f"Could not load model at startup: {e}. /predict will fail until model is available.")
    yield
    pipeline = None


app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="Production-grade sentiment classification powered by fine-tuned BERT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_pipeline():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH.")
    return pipeline


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    """Classify the sentiment of a single text."""
    pipe = _require_pipeline()
    t0 = time.perf_counter()
    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pipe.predict([req.text], return_attention=req.return_attention)
    )
    latency = round((time.perf_counter() - t0) * 1000, 2)
    latency_tracker.record(latency)
    counter.increment()
    return PredictResponse(result=SentimentResult(**results[0]), latency_ms=latency)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(req: BatchPredictRequest):
    """Classify sentiment for a batch of texts (max 64)."""
    pipe = _require_pipeline()
    t0 = time.perf_counter()
    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: pipe.predict(req.texts, return_attention=req.return_attention)
    )
    latency = round((time.perf_counter() - t0) * 1000, 2)
    latency_tracker.record(latency)
    counter.increment(len(req.texts))
    return BatchPredictResponse(
        results=[SentimentResult(**r) for r in results],
        latency_ms=latency,
        count=len(results),
    )


@app.get("/predict/stream", tags=["Inference"])
async def predict_stream(text: str = Query(..., min_length=1, max_length=5000)):
    """Stream word-by-word sentiment analysis via Server-Sent Events."""
    pipe = _require_pipeline()

    async def event_generator() -> AsyncGenerator[str, None]:
        words = text.split()
        accumulated = ""
        for word in words:
            accumulated = (accumulated + " " + word).strip()
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda t=accumulated: pipe.predict([t])[0]
            )
            counter.increment()
            data = json.dumps({
                "word": word,
                "accumulated": accumulated,
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.05)
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Check service health and model status."""
    device = str(getattr(getattr(pipeline, "device", None), "__str__", lambda: "unknown")())
    return HealthResponse(
        status="healthy" if pipeline else "degraded",
        model_loaded=pipeline is not None,
        device=device if pipeline else "none",
        uptime_seconds=counter.uptime_seconds,
        total_predictions=counter.total,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Operations"])
async def metrics():
    """Return latency percentiles and throughput stats."""
    return MetricsResponse(
        latency=latency_tracker.stats(),
        total_predictions=counter.total,
        uptime_seconds=counter.uptime_seconds,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "BERT Sentiment API", "docs": "/docs", "health": "/health"}
