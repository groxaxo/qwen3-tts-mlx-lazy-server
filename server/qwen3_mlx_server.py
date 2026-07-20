"""Lazy resident server for Qwen3-TTS 12Hz MLX (Lucía voice clone).

Holds the model warm so per-sentence latency drops from ~6.5s (cold process)
to ~2-3s. Self-terminates after QWEN3_MLX_TTL_S seconds idle (default 600 =
10 min) so it never lingers — also exits well before the idle-bigmem reaper's
30-min window. Started on demand by tts.sh's qwen3-mlx engine.

Endpoints:
  GET  /health          -> {"ok": true}
  POST /synth {"text", "lang_code", "ref_audio"}  -> audio/wav bytes
"""

import json
import os
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

mx.set_cache_limit(0)
mx.set_memory_limit(int(float(os.environ.get("QWEN3_MLX_MEM_LIMIT_GB", "4")) * 1024**3))

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

HOME = os.path.expanduser("~")
MODEL_PATH = os.environ.get(
    "QWEN3_MLX_MODEL", f"{HOME}/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit"
)
REF_AUDIO = os.environ.get(
    "QWEN3_MLX_REF_AUDIO",
    f"{HOME}/chatterbox-finetunino/latam_runs/profiles/lucia-latam-ar-recipe-ordered/reference.wav",
)
MAX_TOKENS = int(os.environ.get("QWEN3_MLX_MAX_TOKENS", "300"))
TTL_S = int(os.environ.get("QWEN3_MLX_TTL_S", "600"))
PORT = int(os.environ.get("QWEN3_MLX_PORT", "18885"))

sidecar = os.path.splitext(REF_AUDIO)[0] + ".reftext"
REF_TEXT = None
if os.path.isfile(sidecar):
    with open(sidecar) as f:
        REF_TEXT = f.read().strip() or None

print(f"[qwen3-mlx-server] loading {MODEL_PATH}", flush=True)
MODEL = load_model(MODEL_PATH)
print(f"[qwen3-mlx-server] ready on :{PORT}, ttl={TTL_S}s", flush=True)

LAST_USED = time.time()
LOCK = threading.Lock()


def watchdog():
    while True:
        time.sleep(30)
        if time.time() - LAST_USED > TTL_S:
            print("[qwen3-mlx-server] idle TTL reached, exiting", flush=True)
            os._exit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global LAST_USED
        if self.path != "/synth":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            text = payload["text"]
            lang_code = payload["lang_code"]
            ref_audio = payload.get("ref_audio", REF_AUDIO)
            if not isinstance(ref_audio, str):
                raise ValueError("ref_audio must be a path string")
            ref_text = None
            if os.path.isfile(ref_audio):
                sidecar = os.path.splitext(ref_audio)[0] + ".reftext"
                if os.path.isfile(sidecar):
                    with open(sidecar) as f:
                        ref_text = f.read().strip() or None
        except Exception as e:
            self._json(400, {"error": f"bad request: {e}"})
            return
        LAST_USED = time.time()
        try:
            with LOCK, tempfile.TemporaryDirectory() as td:
                prefix = f"synth-{uuid.uuid4().hex[:8]}"
                generate_audio(
                    text,
                    model=MODEL,
                    lang_code=lang_code,
                    ref_audio=ref_audio if os.path.isfile(ref_audio) else None,
                    ref_text=ref_text if os.path.isfile(ref_audio) else None,
                    max_tokens=MAX_TOKENS,
                    output_path=td,
                    file_prefix=prefix,
                    join_audio=True,
                    verbose=False,
                    play=False,
                )
                with open(os.path.join(td, f"{prefix}.wav"), "rb") as f:
                    wav = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        except Exception as e:
            self._json(500, {"error": str(e)})
        finally:
            LAST_USED = time.time()


threading.Thread(target=watchdog, daemon=True).start()
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
