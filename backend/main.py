from __future__ import annotations

import importlib
import os
from typing import List
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

APP_NAME = "CST Hub Backend (Monolith)"
APP_VERSION = "1.0.0"

# -------------------------------------------------------------------
# Create app
# -------------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    description="One FastAPI app exposing 8 project APIs under /api/p1 ... /api/p8",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -------------------------------------------------------------------
# CORS (Vercel rewrites front this, but you can restrict if desired)
# Set FRONTEND_ORIGIN to your Vercel URL to lock this down.
# -------------------------------------------------------------------
FRONTEND = os.getenv("FRONTEND_ORIGIN")  # e.g. "https://your-app.vercel.app"
allow_origins = ["*"] if not FRONTEND else [FRONTEND]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers: include routers defensively (some may not exist yet)
# Each router file must define: router = APIRouter(prefix="/api/pX", ...)
# -------------------------------------------------------------------
def try_include(router_module: str, mounted: List[str]) -> None:
    """
    Attempt to import routers.pX and include its router.
    Collects a list of mounted prefixes for info/debug.
    """
    try:
        mod = importlib.import_module(router_module)
        router = getattr(mod, "router", None)
        if router is None:
            print(f"[main] {router_module} has no 'router' attribute; skipping.")
            return
        app.include_router(router)
        # Try to reveal its prefix for logging
        prefix = getattr(router, "prefix", "<unknown>")
        mounted.append(prefix)
        print(f"[main] Mounted router {router_module} at prefix {prefix}")
    except ModuleNotFoundError:
        print(f"[main] Router module not found: {router_module} (ok to skip)")
    except Exception as e:
        print(f"[main] Failed to mount {router_module}: {e}")

mounted_prefixes: List[str] = []
for mod_name in ("routers.p1", "routers.p2", "routers.p3", "routers.p4",
                 "routers.p5", "routers.p6", "routers.p7", "routers.p8"):
    try_include(mod_name, mounted_prefixes)

# -------------------------------------------------------------------
# Human-friendly root page
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items = "".join(f"<li><code>{p}</code></li>" for p in mounted_prefixes) or "<li>(none)</li>"
    return f"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP_NAME}</title>
<style>
  :root {{ --bg:#0b0c10; --card:#111218; --fg:#e5e7eb; --muted:#9ca3af; --accent:#60a5fa; }}
  body {{ margin:0; font:16px system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial; background:var(--bg); color:var(--fg); }}
  .wrap {{ max-width:880px; margin:0 auto; padding:28px; }}
  .card {{ background:var(--card); border:1px solid #1f2430; border-radius:14px; padding:18px 20px; }}
  h1 {{ margin:0 0 10px; font-size:22px }}
  .muted {{ color:var(--muted); }}
  a {{ color:var(--accent); text-decoration:none; }}
  code {{ background:#0f1117; padding:.2em .35em; border-radius:6px; }}
  ul {{ margin:8px 0 0 18px; }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
<div class="wrap">
  <div class="card">
    <h1>{APP_NAME}</h1>
    <div class="muted">version <b>{APP_VERSION}</b> · {now}</div>
    <div class="grid" style="margin-top:14px">
      <div>
        <h3 style="margin:8px 0">Mounted APIs</h3>
        <ul>{items}</ul>
      </div>
      <div>
        <h3 style="margin:8px 0">Useful links</h3>
        <ul>
          <li><a href="/docs">/docs</a> (Swagger)</li>
          <li><a href="/api/health">/api/health</a></li>
          <li><a href="/api/meta">/api/meta</a></li>
        </ul>
      </div>
    </div>
  </div>
</div>
</html>
"""

# -------------------------------------------------------------------
# JSON meta + health
# -------------------------------------------------------------------
@app.get("/api/meta")
def api_meta():
    """Structured JSON about this deployment."""
    return JSONResponse({
        "ok": True,
        "service": "cst-hub-monolith",
        "name": APP_NAME,
        "version": APP_VERSION,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mounted": mounted_prefixes,
        "links": {
            "docs": "/docs",
            "health": "/api/health"
        }
    })

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cst-hub-monolith", "mounted": mounted_prefixes}

# -------------------------------------------------------------------
# Optional: Local run
# Railway runs: python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
