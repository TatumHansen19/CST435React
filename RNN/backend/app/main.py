"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
from pathlib import Path
import traceback
import json

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
)

# -------------------------------------------------------------------
# App + CORS  (✅ FULLY FIXED FOR VERCEL + RAILWAY)
# -------------------------------------------------------------------
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = [
    "https://cst-435-react.vercel.app",
    "https://cst-435-react-git-main-tatums-projects-965c11b1.vercel.app",
    "https://cst-435-react-n8pzza1hs-tatums-projects-965c11b1.vercel.app",
]

ORIGIN_REGEX = r"^https://cst-435-react(?:-[a-z0-9]+)?-tatums-projects-965c11b1\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# All endpoints live under /api so Vercel rewrite works
api = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# Model Setup
# -------------------------------------------------------------------
generator: TextGenerator | None = None

HERE = Path(__file__).parent
MODEL_DIR = (HERE / "saved_models").resolve()

@app.on_event("startup")
async def load_model():
    global generator
    try:
        print("[BOOT] Expected model directory:", MODEL_DIR)
        model_pt = MODEL_DIR / "model.pt"
        model_h5 = MODEL_DIR / "model.h5"
        cfg_path = MODEL_DIR / "config.json"
        tok_json = MODEL_DIR / "tokenizer.json"
        tok_pkl  = MODEL_DIR / "tokenizer.pkl"

        print("[BOOT] Exists?",
              "model.pt:", model_pt.exists(),
              "| model.h5:", model_h5.exists(),
              "| config.json:", cfg_path.exists(),
              "| tokenizer.json:", tok_json.exists(),
              "| tokenizer.pkl:", tok_pkl.exists())

        if not cfg_path.exists():
            raise FileNotFoundError("config.json missing → Model cannot load")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        generator = TextGenerator(
            sequence_length=cfg.get("sequence_length", 30),
            embedding_dim=cfg.get("embedding_dim", 50),
            lstm_units=cfg.get("lstm_units", 75),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            vocab_size=cfg.get("vocab_size"),
        )
        generator.config = cfg

        print(f"[BOOT] Loading model from {MODEL_DIR} …")
        generator.load_model(str(MODEL_DIR))

        if getattr(generator, "model", None) is None and getattr(generator, "torch_model", None) is None:
            raise RuntimeError("Model object missing after load → check TextGenerator.load_model()")

        print("✅ MODEL LOADED SUCCESSFULLY")

    except Exception as e:
        print("❌ MODEL LOAD FAILED")
        print(traceback.format_exc())

# -------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse)
async def health():
    ready = (
        generator is not None and
        (getattr(generator, "model", None) or getattr(generator, "torch_model", None))
    )
    return HealthResponse(status="healthy" if ready else "model_not_loaded", model_loaded=ready)

@api.get("/model-info", response_model=ModelInfo)
async def model_info():
    if generator is None or generator.config is None:
        raise HTTPException(503, "Model not loaded")
    cfg = generator.config
    return ModelInfo(
        vocabulary_size=cfg["vocab_size"],
        sequence_length=cfg["sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg["lstm_units"],
        num_layers=cfg["num_lstm_layers"],
        is_loaded=True,
    )

@api.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = generator.generate_text(
        seed_text=request.seed_text,
        num_words=request.num_words,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        use_beam_search=request.use_beam_search,
        beam_width=request.beam_width,
    )
    return GenerateResponse(
        seed_text=request.seed_text,
        generated_text=text,
        num_words_generated=max(0, len(text.split()) - len(request.seed_text.split())),
        temperature=request.temperature,
    )

app.include_router(api)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RNN API Ready", "docs": "/docs"}

@app.exception_handler(Exception)
async def handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
