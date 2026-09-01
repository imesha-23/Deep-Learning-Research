#!/usr/bin/env python3
"""
api.py — FastAPI web service for Sinhala Fake News Detection (FastText version)

Uses the same model as cli2.py. Auto-trains on first run.

Run:
    pip install fastapi uvicorn
    python api.py
    # → http://localhost:8000/docs  (Swagger UI)
    # → http://localhost:8000/redoc (ReDoc)
"""

import sys, json
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Import shared logic from cli2.py ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import cli2


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models (shown in Swagger UI)
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Sinhala news article text")


class PredictResponse(BaseModel):
    label: str = Field(description="Prediction: REAL or FAKE")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0–1)")
    probability_fake: float = Field(ge=0.0, le=1.0, description="Probability of FAKE class")
    probability_real: float = Field(ge=0.0, le=1.0, description="Probability of REAL class")
    cleaned_text: str = Field(description="Text after cleaning/preprocessing")


class BatchPredictRequest(BaseModel):
    articles: list[PredictRequest] = Field(min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model: str
    accuracy: float | None = None
    macro_f1: float | None = None
    num_samples: int | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Global model handle (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────
class ModelHolder:
    ft_model = None
    classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup. Train first if artifacts don't exist."""
    artifacts_exist = all(p.exists() for p in [
        cli2.FT_MODEL_PATH, cli2.CLASSIFIER_PATH,
        cli2.VECTORIZER_PATH, cli2.META_PATH,
    ])

    if not artifacts_exist:
        print("[api] First run — training FastText model...", file=sys.stderr)
        cli2.train()
        print(file=sys.stderr)

    ModelHolder.ft_model, ModelHolder.classifier = cli2.load_pipeline()
    print("[api] Model loaded. Ready for requests.", file=sys.stderr)
    yield
    # Cleanup (nothing to do)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sinhala Fake News Detector",
    description=(
        "Classify Sinhala news articles as **REAL** or **FAKE** "
        "using FastText embeddings + Logistic Regression.\n\n"
        "- Model: FastText (100d) + Logistic Regression\n"
        "- No GPU required\n"
        "- First request triggers model training (~30s)"
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Research Project — Sinhala Fake News Detection",
    },
)

# ── CORS (allow browser-based clients) ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Sinhala Fake News Detector",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the service and model are ready."""
    if ModelHolder.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    meta = {}
    if cli2.META_PATH.exists():
        with open(cli2.META_PATH) as f:
            meta = json.load(f)

    return HealthResponse(
        status="ok",
        model=meta.get("architecture", "FastText + LogisticRegression"),
        accuracy=meta.get("accuracy"),
        macro_f1=meta.get("macro_f1"),
        num_samples=meta.get("num_samples"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    """
    Predict whether a Sinhala news article is REAL or FAKE.

    Accepts raw Sinhala text and returns the prediction with confidence scores.
    """
    if ModelHolder.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    cleaned = cli2.clean_text(req.text)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    tokens = cli2.tokenize_and_filter(cleaned)
    vec = cli2.doc_vector(tokens, ModelHolder.ft_model).reshape(1, -1)

    prob = ModelHolder.classifier.predict_proba(vec)[0]
    pred = ModelHolder.classifier.predict(vec)[0]

    return PredictResponse(
        label="REAL" if pred == 1 else "FAKE",
        confidence=float(prob[pred]),
        probability_fake=round(float(prob[0]), 6),
        probability_real=round(float(prob[1]), 6),
        cleaned_text=cleaned,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(req: BatchPredictRequest):
    """
    Batch predict multiple articles (up to 100 at once). Returns predictions
    in the same order as the input.
    """
    if ModelHolder.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    results = []
    for article in req.articles:
        cleaned = cli2.clean_text(article.text)
        if not cleaned.strip():
            results.append(PredictResponse(
                label="UNKNOWN",
                confidence=0.0,
                probability_fake=0.5,
                probability_real=0.5,
                cleaned_text="",
            ))
            continue

        tokens = cli2.tokenize_and_filter(cleaned)
        vec = cli2.doc_vector(tokens, ModelHolder.ft_model).reshape(1, -1)
        prob = ModelHolder.classifier.predict_proba(vec)[0]
        pred = ModelHolder.classifier.predict(vec)[0]

        results.append(PredictResponse(
            label="REAL" if pred == 1 else "FAKE",
            confidence=float(prob[pred]),
            probability_fake=round(float(prob[0]), 6),
            probability_real=round(float(prob[1]), 6),
            cleaned_text=cleaned,
        ))

    return BatchPredictResponse(results=results, total=len(results))


@app.post("/retrain", tags=["System"])
async def retrain():
    """Force retrain the model from scratch."""
    import shutil

    if cli2.ARTIFACT_DIR.exists():
        shutil.rmtree(cli2.ARTIFACT_DIR)

    cli2.train()
    ModelHolder.ft_model, ModelHolder.classifier = cli2.load_pipeline()

    return {
        "status": "ok",
        "message": "Model retrained and reloaded successfully",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )