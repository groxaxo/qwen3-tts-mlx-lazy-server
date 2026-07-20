#!/usr/bin/env bash
# Setup for Qwen3-TTS 12Hz 1.7B Base MLX 8-bit with mlx_audio + lazy server.
#
# The aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit repo targets speech-swift and
# needs two fixes to load in mlx_audio (>=0.4.5):
#   1. config.json lacks speaker_encoder_config -> mlx_audio defaults enc_dim
#      to 1024 and fails with a shape error on speaker_encoder.fc.weight;
#      the 1.7B needs enc_dim 2048.
#   2. It ships no speech_tokenizer/ (Mimi decoder) -> "Speech tokenizer not
#      loaded"; graft it from mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
#      (quantization-independent).
# Optionally writes a .reftext sidecar next to your reference audio so the
# engine skips mlx_audio's in-process Whisper transcription of the reference.
set -euo pipefail

PYTHON="${QWEN3_MLX_PYTHON:?set QWEN3_MLX_PYTHON to a venv python with mlx-audio installed}"
DEST="${QWEN3_MLX_MODEL:-$HOME/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit}"
SRC_REPO="aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit"
TOK_REPO="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

echo "==> downloading $SRC_REPO"
SNAP=$("$PYTHON" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$SRC_REPO'))")
mkdir -p "$DEST"
cp -RL "$SNAP"/. "$DEST"/

echo "==> patching speaker_encoder_config (enc_dim 2048)"
"$PYTHON" - "$DEST/config.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c.setdefault("speaker_encoder_config", {"enc_dim": 2048, "sample_rate": 24000})
json.dump(c, open(p, "w"), indent=2)
PY

echo "==> grafting speech_tokenizer from $TOK_REPO"
"$PYTHON" - "$DEST" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
dest = sys.argv[1]
for f in ["speech_tokenizer/config.json", "speech_tokenizer/configuration.json",
          "speech_tokenizer/model.safetensors", "speech_tokenizer/preprocessor_config.json",
          "generation_config.json", "preprocessor_config.json"]:
    p = hf_hub_download("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", f)
    out = os.path.join(dest, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(p, out)
    print("   ", f)
PY

if [ -n "${QWEN3_MLX_REF_AUDIO:-}" ] && [ -f "$QWEN3_MLX_REF_AUDIO" ]; then
    SIDE="${QWEN3_MLX_REF_AUDIO%.*}.reftext"
    if [ ! -f "$SIDE" ] && curl -s -m 2 -o /dev/null http://127.0.0.1:5093/v1/audio/transcriptions 2>/dev/null; then
        echo "==> transcribing reference for .reftext sidecar (local ASR :5093)"
        curl -s -m 60 -F "file=@$QWEN3_MLX_REF_AUDIO" -F model=parakeet-tdt-0.6b-v3 \
            http://127.0.0.1:5093/v1/audio/transcriptions \
            | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['text'], end='')" > "$SIDE"
        echo "    wrote $SIDE"
    fi
fi

echo "==> smoke test"
TMP=$(mktemp -d)
"$PYTHON" -m mlx_audio.tts.generate --model "$DEST" --text 'Setup complete.' \
    --output_path "$TMP" --file_prefix smoke --audio_format wav --join_audio >/dev/null
[ -s "$TMP/smoke.wav" ] && echo "OK: $DEST" || { echo "FAILED"; exit 1; }
