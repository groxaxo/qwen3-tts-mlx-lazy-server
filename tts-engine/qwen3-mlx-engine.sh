# --- Qwen3-TTS 12Hz MLX (local, Apple Silicon, in-process; clones ref voice) -
: "${QWEN3_MLX_PYTHON:=$CHATTERBOX_TURBO_MLX_PYTHON}"
: "${QWEN3_MLX_MODEL:=$HOME/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit}"
: "${QWEN3_MLX_REF_AUDIO:=$HOME/chatterbox-finetunino/latam_runs/profiles/lucia-latam-ar-recipe-ordered/reference.wav}"
: "${QWEN3_MLX_REF_AUDIO_EN:=$HOME/voices/qwen3-mlx-carina-en.wav}"
: "${QWEN3_MLX_REF_AUDIO_ES:=$HOME/voices/qwen3-mlx-carina-es.wav}"

# Memory: MLX buffer-cache retention spikes generation to ~13GB footprint on
# this model; cache_limit(0) + a relaxed memory_limit caps it at ~6GB with no
# speed or quality loss (ASR-verified 2026-07-20). A .reftext sidecar next to
# the ref audio skips the in-process Whisper transcription of the reference.
: "${QWEN3_MLX_MEM_LIMIT_GB:=4}"
: "${QWEN3_MLX_MAX_TOKENS:=300}"
# Lazy resident server: first call spawns it (~10s), later calls are ~2-3s;
# it exits by itself after QWEN3_MLX_TTL_S idle (default 15 min).
: "${QWEN3_MLX_LAZY:=1}"
: "${QWEN3_MLX_PORT:=18885}"
: "${QWEN3_MLX_TTL_S:=900}"
: "${QWEN3_MLX_SERVER:=$HOME/.config/opencode/qwen3_mlx_server.py}"

_qwen3_mlx_server_synth() {
    local text="$1"
    local lang="$2"
    local out_wav="$3"
    local url="http://127.0.0.1:${QWEN3_MLX_PORT}"

    if ! curl -s -m 2 -o /dev/null "$url/health"; then
        [ -f "$QWEN3_MLX_SERVER" ] || return 1
        echo "[tts] Qwen3-TTS MLX spawning lazy server on :${QWEN3_MLX_PORT}…" >&2
        QWEN3_MLX_MODEL="$QWEN3_MLX_MODEL" QWEN3_MLX_REF_AUDIO="$QWEN3_MLX_REF_AUDIO" \
        QWEN3_MLX_MEM_LIMIT_GB="$QWEN3_MLX_MEM_LIMIT_GB" QWEN3_MLX_MAX_TOKENS="$QWEN3_MLX_MAX_TOKENS" \
        QWEN3_MLX_TTL_S="$QWEN3_MLX_TTL_S" QWEN3_MLX_PORT="$QWEN3_MLX_PORT" \
        nohup "$QWEN3_MLX_PYTHON" "$QWEN3_MLX_SERVER" >> "$HOME/Library/Logs/qwen3-mlx-server.log" 2>&1 &
        local i=0
        while [ $i -lt 45 ]; do
            curl -s -m 2 -o /dev/null "$url/health" && break
            sleep 1; i=$((i + 1))
        done
        curl -s -m 2 -o /dev/null "$url/health" || {
            echo "tts.sh: Qwen3-TTS MLX lazy server failed to start" >&2
            return 1
        }
    fi

    local payload
    payload=$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1], "lang_code": sys.argv[2], "ref_audio": sys.argv[3]}))' "$text" "$lang" "$QWEN3_MLX_REF_AUDIO") || return 1
    curl -s -m 120 -X POST "$url/synth" -H 'Content-Type: application/json' \
        -d "$payload" -o "$out_wav" || return 1
    [ -s "$out_wav" ] && head -c 4 "$out_wav" | grep -q 'RIFF' || {
        echo "tts.sh: Qwen3-TTS MLX server returned no audio" >&2
        rm -f "$out_wav"
        return 1
    }
    return 0
}

_synth_qwen3_mlx() {
    local text="$1"
    local lang="$2"
    local out_wav="$3"
    local output_dir output_name ref_text=""

    output_dir=$(dirname "$out_wav")
    output_name=$(basename "$out_wav" .wav)
    [ -f "${QWEN3_MLX_REF_AUDIO%.*}.reftext" ] && ref_text=$(cat "${QWEN3_MLX_REF_AUDIO%.*}.reftext")
    echo "[tts] Qwen3-TTS MLX lang=${lang} model=${QWEN3_MLX_MODEL} ref=$(basename "$QWEN3_MLX_REF_AUDIO") memcap=${QWEN3_MLX_MEM_LIMIT_GB}GB" >&2

    if [ "$QWEN3_MLX_LAZY" = "1" ] && _qwen3_mlx_server_synth "$text" "$lang" "$out_wav"; then
        fade_wav_edges "$out_wav"
        return 0
    fi
    [ "$QWEN3_MLX_LAZY" = "1" ] && echo "[tts] Qwen3-TTS MLX server route failed → one-shot in-process" >&2

    TTS_TEXT="$text" QWEN3_OUT_DIR="$output_dir" QWEN3_OUT_NAME="$output_name" \
    QWEN3_LANG_CODE="$lang" \
    QWEN3_REF_TEXT="$ref_text" QWEN3_MLX_MODEL="$QWEN3_MLX_MODEL" \
    QWEN3_MLX_REF_AUDIO="$QWEN3_MLX_REF_AUDIO" \
    QWEN3_MLX_MEM_LIMIT_GB="$QWEN3_MLX_MEM_LIMIT_GB" \
    QWEN3_MLX_MAX_TOKENS="$QWEN3_MLX_MAX_TOKENS" \
    "$QWEN3_MLX_PYTHON" -c '
import os
import mlx.core as mx
mx.set_cache_limit(0)
mx.set_memory_limit(int(float(os.environ["QWEN3_MLX_MEM_LIMIT_GB"]) * 1024**3))
from mlx_audio.tts.generate import generate_audio
ref_text = os.environ.get("QWEN3_REF_TEXT") or None
ref_audio = os.environ.get("QWEN3_MLX_REF_AUDIO") or None
if ref_audio and not os.path.isfile(ref_audio):
    ref_audio = None
generate_audio(
    os.environ["TTS_TEXT"],
    model=os.environ["QWEN3_MLX_MODEL"],
    lang_code=os.environ["QWEN3_LANG_CODE"],
    ref_audio=ref_audio,
    ref_text=ref_text if ref_audio else None,
    max_tokens=int(os.environ["QWEN3_MLX_MAX_TOKENS"]),
    output_path=os.environ["QWEN3_OUT_DIR"],
    file_prefix=os.environ["QWEN3_OUT_NAME"],
    join_audio=True, verbose=False, play=False,
)' >/dev/null || return 1
    [ -f "$out_wav" ] && [ -s "$out_wav" ] || {
        echo "tts.sh: Qwen3-TTS MLX produced no audio" >&2
        return 1
    }
    fade_wav_edges "$out_wav"
}

speak_qwen3_mlx() {
    local text="$1"
    local lang="$2"
    local reference="$QWEN3_MLX_REF_AUDIO"

    case "$lang" in
        en*) [ -f "$QWEN3_MLX_REF_AUDIO_EN" ] && reference="$QWEN3_MLX_REF_AUDIO_EN" ;;
        es*) [ -f "$QWEN3_MLX_REF_AUDIO_ES" ] && reference="$QWEN3_MLX_REF_AUDIO_ES" ;;
    esac

    [ "$(uname -s 2>/dev/null)" = "Darwin" ] || return 1
    [ -x "$QWEN3_MLX_PYTHON" ] || {
        echo "tts.sh: Qwen3-TTS MLX Python not found: $QWEN3_MLX_PYTHON" >&2
        return 1
    }
    [ -d "$QWEN3_MLX_MODEL" ] || {
        echo "tts.sh: Qwen3-TTS MLX model not found: $QWEN3_MLX_MODEL" >&2
        return 1
    }

    rm -f "$OUTPUT"
    QWEN3_MLX_REF_AUDIO="$reference" _synth_qwen3_mlx "$text" "$lang" "$OUTPUT" || return 1
    if [ "${TTS_NO_PLAY:-0}" = "1" ]; then
        echo "$OUTPUT"
        return 0
    fi
    play_wav "$OUTPUT"
    rm -f "$OUTPUT"
}
