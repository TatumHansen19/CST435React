"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
from pathlib import Path
import time

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
# App + CORS
# -------------------------------------------------------------------
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

FRONTEND = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND] if FRONTEND != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All public routes live under /api so the Vercel rewrite works
api = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------
generator: TextGenerator | None = None

_current_dir = Path(__file__).parent
_backend_dir = _current_dir.parent if _current_dir.name == "app" else _current_dir
MODEL_PATH = _backend_dir / "saved_models" / "model.h5"
TOKENIZER_PATH = _backend_dir / "saved_models" / "tokenizer.pkl"
CONFIG_PATH = _backend_dir / "saved_models" / "config.json"


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global generator
    try:
        possible_model_dirs = [
            str(MODEL_PATH.parent),   # saved_models (absolute)
            "saved_models",           # relative from app/
            "../saved_models",        # parent of app/
        ]

        for model_dir in possible_model_dirs:
            model_h5 = Path(model_dir) / "model.h5"
            model_pt = Path(model_dir) / "model.pt"
            config_file = Path(model_dir) / "config.json"

            # Prefer PyTorch .pt if present
            if model_pt.exists():
                loaded_config = None
                if config_file.exists():
                    import json
                    with open(config_file, "r") as f:
                        loaded_config = json.load(f)

                seq_len = loaded_config.get("sequence_length", 30) if loaded_config else 30
                emb_dim = loaded_config.get("embedding_dim", 50) if loaded_config else 50
                lstm_u = loaded_config.get("lstm_units", 75) if loaded_config else 75
                num_layers = loaded_config.get("num_lstm_layers", 1) if loaded_config else 1
                dropout = loaded_config.get("dropout_rate", 0.2) if loaded_config else 0.2
                vocab_size = loaded_config.get("vocab_size") if loaded_config else None

                generator = TextGenerator(
                    sequence_length=seq_len,
                    embedding_dim=emb_dim,
                    lstm_units=lstm_u,
                    num_lstm_layers=num_layers,
                    dropout_rate=dropout,
                    vocab_size=vocab_size,
                )
                if loaded_config:
                    generator.config = loaded_config

                generator.load_model(model_dir)
                print(f"✓ PyTorch model loaded successfully from {model_dir}")
                return

            # Fallback: legacy Keras .h5 with config.json
            if model_h5.exists() and config_file.exists():
                import json
                with open(config_file, "r") as f:
                    loaded_config = json.load(f)

                generator = TextGenerator(
                    sequence_length=loaded_config.get("sequence_length", 30),
                    embedding_dim=loaded_config.get("embedding_dim", 50),
                    lstm_units=loaded_config.get("lstm_units", 75),
                    num_lstm_layers=loaded_config.get("num_lstm_layers", 1),
                    dropout_rate=loaded_config.get("dropout_rate", 0.2),
                    vocab_size=loaded_config.get("vocab_size"),
                )
                generator.config = loaded_config
                generator.load_model(model_dir)
                print(f"✓ Legacy Keras model loaded successfully from {model_dir}")
                return

        print("⚠ Model not found. Please train the model first (python train.py).")

    except Exception as e:
        import traceback
        print(f"✗ Error loading model: {e}")
        print(traceback.format_exc())

# -------------------------------------------------------------------
# API ROUTES (all under /api/*)
# -------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse, tags=["Health"])
async def api_health():
    is_loaded = generator is not None and getattr(generator, "model", None) is not None
    return HealthResponse(
        status="healthy" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
    )

@api.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")

    cfg = generator.config
    return ModelInfo(
        vocabulary_size=cfg["vocab_size"],
        sequence_length=cfg["sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg["lstm_units"],
        num_layers=cfg["num_lstm_layers"],
        is_loaded=True,
    )

@api.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width,
        )
        num_generated = max(0, len(generated.split()) - len(request.seed_text.split()))
        return GenerateResponse(
            seed_text=request.seed_text,
            generated_text=generated,
            num_words_generated=num_generated,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")

@api.get("/stats", tags=["Stats"])
async def get_stats():
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")

    stats = {
        "model_config": generator.config,
        "model_parameters": generator.count_params(),
        "training_history_available": generator.history is not None,
    }
    if generator.history is not None:
        h = generator.history.history
        stats["training_stats"] = {
            "final_loss": float(h["loss"][-1]),
            "final_val_loss": float(h["val_loss"][-1]),
            "epochs_trained": len(h["loss"]),
        }
    return stats

@api.get("/visualizations/architecture", tags=["Visualizations"])
async def get_architecture():
    path = "visualizations/model_architecture.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")
    return FileResponse(path=path, media_type="image/png")

@api.get("/visualizations/training", tags=["Visualizations"])
async def get_training_history():
    path = "visualizations/training_history.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training history plot not found.")
    return FileResponse(path=path, media_type="image/png")

app.include_router(api)

# -------------------------------------------------------------------
# Root + error handler
# -------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "RNN Text Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_prefix": "/api",
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
