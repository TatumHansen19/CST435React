from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional, Dict, Any
import os, json, importlib

# ---- Import your Pydantic models (exactly as you provided) ----
P5_MODELS = importlib.import_module("projects.p5.models")
GenerateRequest = getattr(P5_MODELS, "GenerateRequest")
GenerateResponse = getattr(P5_MODELS, "GenerateResponse")
ModelInfo = getattr(P5_MODELS, "ModelInfo")
HealthResponse = getattr(P5_MODELS, "HealthResponse")

router = APIRouter(prefix="/api/p5", tags=["p5 (RNN)"])

# Lazy singletons
_generator = None
_cfg: Optional[Dict[str, Any]] = None
_MODEL_DIR: Optional[Path] = None


# --------------------------- Model dir helpers ---------------------------
def _candidate_model_dirs():
    env_dir = os.getenv("MODEL_DIR") or os.getenv("RNN_MODEL_DIR")
    here = Path(__file__).resolve()
    out = []
    if env_dir:
        out.append(Path(env_dir).expanduser().resolve())
    out += [
        here.parents[2] / "projects" / "p5" / "saved_models",
        Path.cwd() / "saved_models",
    ]
    # de-dupe
    seen, uniq = set(), []
    for p in (d.resolve() for d in out):
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq

def _pick_model_dir() -> Optional[Path]:
    for d in _candidate_model_dirs():
        if d.is_dir() and (d / "config.json").exists() and (d / "tokenizer.json").exists():
            return d
    return None

def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


# ------------------------------ Loader -----------------------------------
def _ensure_loaded():
    """
    Lazy-loads config, tokenizer, and model weights using your TextGenerator.
    Assumes model.pt is a PyTorch state_dict (as your save_model() writes).
    """
    global _generator, _cfg, _MODEL_DIR
    if _generator is not None:
        return

    _MODEL_DIR = _pick_model_dir()
    if _MODEL_DIR is None:
        raise HTTPException(503, "Model directory not found (need config.json + tokenizer.json).")

    cfg_path = _MODEL_DIR / "config.json"
    weights_path = _MODEL_DIR / "model.pt"

    if not cfg_path.exists():
        raise HTTPException(503, f"Missing config.json at {_MODEL_DIR}")
    if not weights_path.exists():
        raise HTTPException(503, f"Missing model.pt at {_MODEL_DIR}")
    if _is_lfs_pointer(weights_path):
        raise HTTPException(500, "model.pt is a Git LFS pointer; run `git lfs pull` in deploy.")

    _cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Import your generator only when needed (keeps cold starts lighter)
    from projects.p5.text_generator import TextGenerator

    _generator = TextGenerator(
        sequence_length=_cfg.get("sequence_length", 50),
        embedding_dim=_cfg.get("embedding_dim", 100),
        lstm_units=_cfg.get("lstm_units", 150),
        num_lstm_layers=_cfg.get("num_lstm_layers", 2),
        dropout_rate=_cfg.get("dropout_rate", 0.2),
        # You also support these — keep defaults unless you store in config.json:
        recurrent_dropout=_cfg.get("recurrent_dropout", 0.0),
        activation_fn=_cfg.get("activation_fn", "relu"),
        use_glove_embeddings=_cfg.get("use_glove_embeddings", False),
        trainable_embeddings=_cfg.get("trainable_embeddings", True),
        vocab_size=_cfg.get("vocab_size"),
    )

    # This calls your loader which:
    #  - reads config.json (again) to refresh hyperparams
    #  - loads tokenizer.json / tokenizer.pkl
    #  - builds LSTMModel and loads state_dict from model.pt
    _generator.load_model(str(_MODEL_DIR))


# ------------------------------ Routes -----------------------------------
@router.get("/health", response_model=HealthResponse)
def health():
    try:
        _ensure_loaded()
        return HealthResponse(status="healthy", model_loaded=True)
    except HTTPException:
        # Don’t leak internal details to the health ping
        return HealthResponse(status="error", model_loaded=False)

@router.get("/model-info", response_model=ModelInfo)
def model_info():
    _ensure_loaded()
    cfg = _cfg or {}
    return ModelInfo(
        vocabulary_size=cfg.get("vocab_size") or 0,
        sequence_length=cfg.get("sequence_length") or 0,
        embedding_dim=cfg.get("embedding_dim") or 0,
        lstm_units=cfg.get("lstm_units") or 0,
        num_layers=cfg.get("num_lstm_layers") or 0,
        is_loaded=True,
    )

@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    _ensure_loaded()

    # Your models.py allows top_k=0 and top_p=1.0 — interpret these as “disable”.
    top_k = None if (request.top_k is not None and request.top_k <= 0) else request.top_k
    top_p = None if (request.top_p is not None and request.top_p >= 1.0) else request.top_p

    # Call your generator with exactly the parameters it supports.
    # You also support repetition_penalty & diversity_boost; keep your defaults unless you add them to the request model.
    text = _generator.generate_text(
        seed_text=request.seed_text,
        num_words=request.num_words,
        temperature=request.temperature,
        top_k=top_k if top_k is not None else 40,
        top_p=top_p if top_p is not None else 0.92,
        use_beam_search=request.use_beam_search,
        beam_width=request.beam_width,
        # repetition_penalty=2.5,
        # diversity_boost=1.0,
    )

    return GenerateResponse(
        seed_text=request.seed_text,
        generated_text=text,
        num_words_generated=max(0, len(text.split()) - len(request.seed_text.split())),
        temperature=request.temperature,
    )
