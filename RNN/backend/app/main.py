"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
from pathlib import Path
import traceback
import json
from typing import Optional

# Ensure local imports work (text_generator.py, models.py live one level up from this file)
HERE = Path(__file__).parent.resolve()            # rnn/backend/app
sys.path.insert(0, str(HERE))                     # for local imports in this folder
sys.path.insert(0, str(HERE.parent))              # rnn/backend

from text_generator import TextGenerator  # noqa: E402
from models import (                      # noqa: E402
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
)

# -------------------------------------------------------------------
# App + CORS
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
    "http://localhost:5173",
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

# All endpoints live under /api so Vercel/Railway rewrites work
api = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# Model directory resolution
# -------------------------------------------------------------------

def _candidate_model_dirs() -> list[Path]:
    """
    Build an ordered list of candidate model directories.
    Priority:
      1) $MODEL_DIR
      2) $RNN_MODEL_DIR
      3) rnn/backend/saved_models  (based on this file's location)
      4) rnn/saved_models          (two levels up)
      5) ./saved_models            (cwd fallback)
    """
    paths: list[Path] = []
    env_model_dir = os.getenv("MODEL_DIR") or os.getenv("RNN_MODEL_DIR")
    if env_model_dir:
        try:
            paths.append(Path(env_model_dir).expanduser().resolve())
        except Exception:
            pass

    # This file is rnn/backend/app/main.py → parents[1] == rnn/backend
    paths += [
        (HERE.parent / "saved_models").resolve(),          # rnn/backend/saved_models
        (HERE.parent.parent / "saved_models").resolve(),   # rnn/saved_models
        (Path.cwd() / "saved_models").resolve(),           # working dir fallback
    ]

    # de-duplicate preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _pick_model_dir() -> Optional[Path]:
    searched = []
    for d in _candidate_model_dirs():
        searched.append(str(d))
        if d.is_dir() and (d / "config.json").exists() and (d / "tokenizer.json").exists():
            print(f"[BOOT] Using model directory: {d}", flush=True)
            return d
    print("[BOOT] Model dir not found. Searched:", searched, flush=True)
    return None


def _is_lfs_pointer(path: Path) -> bool:
    """Best-effort detection for Git LFS pointer files."""
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


MODEL_DIR: Optional[Path] = _pick_model_dir()

# -------------------------------------------------------------------
# Load model on startup
# -------------------------------------------------------------------
generator: Optional[TextGenerator] = None


@app.on_event("startup")
async def load_model() -> None:
    global generator

    print("[BOOT] ENV MODEL_DIR     =", os.getenv("MODEL_DIR"), flush=True)
    print("[BOOT] ENV RNN_MODEL_DIR =", os.getenv("RNN_MODEL_DIR"), flush=True)
    print("[BOOT] Final MODEL_DIR   =", str(MODEL_DIR), flush=True)

    try:
        if MODEL_DIR is None:
            raise FileNotFoundError("No candidate model directory contained required files (config.json, tokenizer.json)")

        model_pt = MODEL_DIR / "model.pt"
        cfg_path = MODEL_DIR / "config.json"
        tok_json = MODEL_DIR / "tokenizer.json"

        print(
            "[BOOT] Exists? ",
            "model.pt:", model_pt.exists(),
            "| config.json:", cfg_path.exists(),
            "| tokenizer.json:", tok_json.exists(),
            flush=True,
        )

        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} missing → Model cannot load")

        # Load config
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Instantiate generator with config
        generator = TextGenerator(
            sequence_length=cfg.get("sequence_length", 30),
            embedding_dim=cfg.get("embedding_dim", 50),
            lstm_units=cfg.get("lstm_units", 75),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            vocab_size=cfg.get("vocab_size"),
        )
        generator.config = cfg

        # Helpful early check for LFS pointer
        if model_pt.exists() and _is_lfs_pointer(model_pt):
            raise RuntimeError(
                f"{model_pt} looks like a Git LFS pointer, not real weights. "
                "Ensure your repo commits the real binary or your build runs "
                "`git lfs install && git lfs fetch --all && git lfs checkout`."
            )

        print(f"[BOOT] Loading model from {MODEL_DIR} …", flush=True)
        generator.load_model(str(MODEL_DIR))

        if getattr(generator, "model", None) is None and getattr(generator, "torch_model", None) is None:
            raise RuntimeError("Model object missing after load → check TextGenerator.load_model()")

        print("✅ MODEL LOADED SUCCESSFULLY", flush=True)

    except Exception as e:
        print("❌ MODEL LOAD FAILED", flush=True)
        print(traceback.format_exc(), flush=True)
        # keep app running so /api/health can report status


# -------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ready = (
        generator is not None and
        (getattr(generator, "model", None) is not None or getattr(generator, "torch_model", None) is not None)
    )
    return HealthResponse(status="healthy" if ready else "model_not_loaded", model_loaded=ready)


@api.get("/model-info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
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


# Compatibility alias if frontend calls /model/info
@app.get("/model/info")
async def model_info_alias():
    return await model_info()


@api.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
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
