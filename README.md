# qwen3-tts-mlx-lazy-server

Local, CPU/GPU-friendly voice cloning on Apple Silicon with
[Qwen3-TTS 12Hz 1.7B Base](https://huggingface.co/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit)
(MLX 8-bit) via [mlx-audio](https://github.com/Blaizzy/mlx-audio), packaged as a
**lazy resident server** with an idle self-shutdown, plus a drop-in `tts.sh`
engine.

Clones any voice from a ~6-15 s reference clip (`--ref_audio`). Verified with a
Parakeet ASR round-trip: transcription of generated Spanish/English audio
matches the input text exactly.

## Why this repo exists

The aufklarer MLX conversion targets `speech-swift` and does not load in
mlx_audio as-is. Three fixes, applied by `setup.sh`:

1. **`speaker_encoder_config` missing** → mlx_audio defaults `enc_dim=1024`,
   the 1.7B checkpoint needs **2048** (shape error on `speaker_encoder.fc.weight`).
2. **No `speech_tokenizer/`** (Mimi decoder) → grafted from
   `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (quantization-independent).
3. **Memory spike and disabled-cache slowdown**: generation originally reached
   ~13 GB with MLX's default cache behavior. Disabling the cache controlled
   memory, but a seeded comparison showed that a bounded **512 MiB Metal
   cache** is about **9.7% faster** than zero cache. The same outputs were
   byte-identical at 0, 512, and 1024 MiB.

Also: a `.reftext` sidecar next to the reference wav feeds `ref_text` and skips
mlx_audio's in-process Whisper transcription of the reference on every call.

## Benchmarks (Apple M5, 24 GB)

RTF is generation time divided by output duration, so lower is faster.

| Warm server setting | Median RTF | Effective speed | Result |
|---|---:|---:|---|
| Cache disabled | 0.664 | 1.51x realtime | baseline |
| **512 MiB cache** | **0.600** | **1.67x realtime** | accepted |
| 1024 MiB cache | 0.617 | 1.62x realtime | no further gain |

The controlled comparison used the same three Spanish phrases and seeds. All
three WAV files were byte-identical across settings. A separate five-phrase
run with the accepted setting measured **0.604 median RTF** (1.66x realtime,
0.581-0.651 range) and friendly WER 0 on all five clips through local Parakeet
Spanish ASR. After that run, process RSS was 2.80 GiB, MLX active memory was
2.97 GiB, and the retained Metal cache was 125 MiB. See
[`benchmarks/apple-m5-2026-07-22.json`](benchmarks/apple-m5-2026-07-22.json).

## Layout

- `server/qwen3_mlx_server.py` — stdlib-only HTTP server on `:18885`
  (`GET /health`, `POST /synth {"text": …}` → `audio/wav`). Loads the model
  once; exits by itself after `QWEN3_MLX_TTL_S` (default 600 s) idle.
- `tts-engine/qwen3-mlx-engine.sh` — the `qwen3-mlx` engine block used in
  `tts.sh`: tries the server (spawning it on demand), falls back to a one-shot
  in-process generation with the same memory caps.
- `setup.sh` — downloads the model, applies the config fixes, grafts the
  speech tokenizer, optionally writes the `.reftext` sidecar, smoke-tests.

## Quick start

To make this exact checkpoint the default in the shared `talk.sh`/`tts.sh`
dispatcher, set both the engine and model explicitly. This avoids a stale shell
override silently selecting another local TTS engine:

```bash
export TTS_ENGINE=qwen3-mlx
export QWEN3_MLX_MODEL=~/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit
export QWEN3_MLX_CACHE_LIMIT_MB=512
export QWEN3_MLX_REF_AUDIO_ES=~/voices/qwen3-mlx-carina-es.wav
export QWEN3_MLX_REF_AUDIO_EN=~/voices/qwen3-mlx-carina-en.wav
```

For a bilingual Carina setup, keep exact `.reftext` transcripts next to both
PCM WAV references. Confirm the effective selection with:

```bash
~/.config/opencode/skills/talk/talk.sh status
```

The engine selects the Spanish reference for `es*` calls and the English
reference for `en*` calls. The lazy server accepts the selected reference and
language on every request, so a warm process does not accidentally reuse the
previous language's voice conditioning.

```bash
export QWEN3_MLX_PYTHON=~/.venvs/mlx-audio/bin/python   # venv with mlx-audio
export QWEN3_MLX_REF_AUDIO=~/voices/my-voice-ref.wav    # optional, for cloning
./setup.sh

# one-shot
$QWEN3_MLX_PYTHON -m mlx_audio.tts.generate \
  --model ~/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit \
  --text 'Hola, probando.' --ref_audio "$QWEN3_MLX_REF_AUDIO" \
  --output_path . --file_prefix out --audio_format wav --join_audio

# resident server
QWEN3_MLX_PYTHON=$QWEN3_MLX_PYTHON python server/qwen3_mlx_server.py &
curl -X POST localhost:18885/synth -d '{"text":"Hola, probando."}' -o out.wav
```

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `QWEN3_MLX_MODEL` | `~/mlx-models/qwen3-tts-12hz-1.7b-base-mlx-8bit` | model dir |
| `QWEN3_MLX_REF_AUDIO` | (Lucía reference) | voice to clone; `.reftext` sidecar skips Whisper |
| `QWEN3_MLX_MEM_LIMIT_GB` | `4` | relaxed MLX memory limit (caps the spike) |
| `QWEN3_MLX_CACHE_LIMIT_MB` | `512` | retained Metal free-buffer cache; measured speed/memory optimum |
| `QWEN3_MLX_MAX_TOKENS` | `300` | ≈24 s audio headroom per call |
| `QWEN3_MLX_PORT` | `18885` | server port |
| `QWEN3_MLX_TTL_S` | `600` | idle seconds before the server exits |
| `QWEN3_MLX_LAZY` | `1` | engine: try/spawn the server before one-shot |

## Notes

- 4-bit: no public 4-bit exists for 1.7B **Base** (only CustomVoice/VoiceDesign
  and the 0.6B). 4-bit would not help anyway — the memory spike was cache, not
  weights — and 4-bit TTS quantization audibly garbles words.
- The 10-min TTL also keeps the server clear of longer idle-process reapers.
