#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HIGGS_APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${HIGGS_VENV:-/root/higgs-audio-v3-env}"
MODEL_DIR="${HIGGS_MODEL_DIR:-/root/models/higgs-audio-v3-tts-4b-transformers}"
OUTPUT_DIR="${HIGGS_OUTPUT_DIR:-$APP_DIR/outputs}"
OUT_FILE="$OUTPUT_DIR/verify-higgs-v3.wav"

mkdir -p "$OUTPUT_DIR"

"$VENV_DIR/bin/python" "$APP_DIR/run_higgs_v3_transformers.py" \
  --model-path "$MODEL_DIR" \
  --text "This is a Higgs Audio v3 install verification." \
  --out "$OUT_FILE" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95

"$VENV_DIR/bin/python" - "$OUT_FILE" <<'PY'
import sys
import soundfile as sf

path = sys.argv[1]
data, sr = sf.read(path, always_2d=True)
duration = data.shape[0] / sr
peak = float(abs(data).max()) if data.size else 0.0
print(f"verified {path}: sr={sr} channels={data.shape[1]} duration={duration:.2f}s peak={peak:.4f}")
PY
