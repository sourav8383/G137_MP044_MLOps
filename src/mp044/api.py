"""Stage 3: FastAPI prediction service.

Run from the repository root:
    uvicorn mp044.api:app --app-dir src --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from mp044.schemas import BatchRequest, BatchResponse, MilkRecord, Prediction
from mp044.serving import ModelService

log = logging.getLogger("mp044.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = ModelService()
    except Exception:  # keep the app up so /health can report the problem
        log.exception("Model could not be loaded")
        app.state.service = None
    yield


app = FastAPI(title="MP044 Village Milk Collection API", version="1.0.0", lifespan=lifespan)


def get_service(request: Request) -> ModelService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return service


@app.get("/health")
def health(request: Request):
    get_service(request)
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request):
    service = get_service(request)
    return {
        "model_file": service.model_path.name,
        "model_sha256": service.model_sha256,
        "features": service.features,
    }


@app.post("/predict", response_model=Prediction)
def predict(record: MilkRecord, request: Request):
    value = get_service(request).predict([record.model_dump()])[0]
    return Prediction(record_id=record.record_id, prediction=value)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    values = get_service(request).predict([r.model_dump() for r in batch.records])
    return BatchResponse(
        predictions=[
            Prediction(record_id=r.record_id, prediction=v)
            for r, v in zip(batch.records, values)
        ]
    )
