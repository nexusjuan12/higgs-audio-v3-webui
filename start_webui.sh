#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HIGGS_APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${HIGGS_VENV:-$APP_DIR/.venv}"
MODEL_ROOT="${HIGGS_MODEL_ROOT:-$APP_DIR/models}"
MODEL_DIR="${HIGGS_MODEL_DIR:-$MODEL_ROOT/higgs-audio-v3-tts-4b-transformers}"
ASR_MODEL_DIR="${HIGGS_ASR_MODEL_DIR:-$MODEL_ROOT/whisper-base}"
VOICE_DIR="${HIGGS_VOICE_DIR:-$APP_DIR/voice_prompts}"
OUTPUT_DIR="${HIGGS_OUTPUT_DIR:-$APP_DIR/outputs}"
HOST="${HIGGS_HOST:-0.0.0.0}"
PORT="${HIGGS_PORT:-7861}"
DTYPE="${HIGGS_DTYPE:-auto}"
ASR_DEVICE="${HIGGS_ASR_DEVICE:-cuda}"
ASR_DTYPE="${HIGGS_ASR_DTYPE:-auto}"
LOG_FILE="${HIGGS_LOG:-$APP_DIR/higgs-audio-v3-webui.log}"
PID_FILE="${HIGGS_PID_FILE:-$APP_DIR/higgs-audio-v3-webui.pid}"
FOREGROUND="${FOREGROUND:-0}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing Python env: $VENV_DIR"
  echo "Run ./install_5090.sh or ./install_p100.sh first."
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing Higgs model: $MODEL_DIR"
  echo "Run ./install_5090.sh or ./install_p100.sh first."
  exit 1
fi

if [[ ! -d "$ASR_MODEL_DIR" ]]; then
  echo "Missing Whisper ASR model: $ASR_MODEL_DIR"
  echo "Run ./install_5090.sh or ./install_p100.sh first."
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Higgs Audio v3 UI already running: pid=$pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
cmd=(
  "$VENV_DIR/bin/python"
  "$APP_DIR/higgs_v3_webui.py"
  --model-path "$MODEL_DIR"
  --asr-model-path "$ASR_MODEL_DIR"
  --voice-dir "$VOICE_DIR"
  --output-dir "$OUTPUT_DIR"
  --server-name "$HOST"
  --server-port "$PORT"
  --dtype "$DTYPE"
  --asr-device "$ASR_DEVICE"
  --asr-dtype "$ASR_DTYPE"
)

if [[ "$FOREGROUND" == "1" ]]; then
  exec "${cmd[@]}"
fi

setsid "${cmd[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "Started Higgs Audio v3 UI: pid=$(cat "$PID_FILE")"
echo "Local URL: http://$HOST:$PORT/"
echo "Log: $LOG_FILE"
