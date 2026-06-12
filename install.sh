#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${HIGGS_APP_DIR:-/root/higgs-audio-v3-webui}"
VENV_DIR="${HIGGS_VENV:-/root/higgs-audio-v3-env}"
MODEL_DIR="${HIGGS_MODEL_DIR:-/root/models/higgs-audio-v3-tts-4b-transformers}"
TOKENIZER_DIR="${HIGGS_TOKENIZER_DIR:-/root/models/higgs-audio-v2-tokenizer}"
VOICE_DIR="${HIGGS_VOICE_DIR:-$APP_DIR/voice_prompts}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu130}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+cu130}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_PYTHON_ENV="${SKIP_PYTHON_ENV:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_VOICE_PROMPTS="${SKIP_VOICE_PROMPTS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_apt_packages() {
  if [[ "$SKIP_APT" == "1" ]]; then
    return
  fi

  if ! need_cmd apt-get; then
    echo "apt-get not found; skipping system package install"
    return
  fi

  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    lsof \
    net-tools \
    rsync \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    rdma-core
}

install_python_env() {
  if [[ "$SKIP_PYTHON_ENV" == "1" ]]; then
    return
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

  if ! "$VENV_DIR/bin/python" -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"; then
    echo "Pinned CUDA torch install failed; falling back to current cu130 nightly packages"
    "$VENV_DIR/bin/python" -m pip install --pre \
      --index-url "$TORCH_INDEX_URL" \
      torch torchaudio
  fi

  "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
}

download_models() {
  if [[ "$SKIP_MODEL_DOWNLOAD" == "1" ]]; then
    return
  fi

  mkdir -p "$(dirname "$MODEL_DIR")" "$(dirname "$TOKENIZER_DIR")"
  "$VENV_DIR/bin/hf" download multimodalart/higgs-audio-v3-tts-4b-transformers \
    --local-dir "$MODEL_DIR"
  "$VENV_DIR/bin/hf" download bosonai/higgs-audio-v2-tokenizer \
    --local-dir "$TOKENIZER_DIR"

  "$VENV_DIR/bin/python" - "$MODEL_DIR/config.json" "$TOKENIZER_DIR" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
tokenizer_dir = str(Path(sys.argv[2]).resolve())
config = json.loads(config_path.read_text(encoding="utf-8"))
config["audio_tokenizer_id"] = tokenizer_dir
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(f"patched {config_path} audio_tokenizer_id -> {tokenizer_dir}")
PY
}

install_app_files() {
  mkdir -p "$APP_DIR" "$VOICE_DIR" "$APP_DIR/outputs"

  if [[ "$(realpath "$SCRIPT_DIR")" != "$(realpath "$APP_DIR")" ]]; then
    rsync -a \
      --exclude ".git" \
      --exclude ".env" \
      --exclude "outputs" \
      --exclude "voice_prompts" \
      "$SCRIPT_DIR/" "$APP_DIR/"
  fi

  chmod +x "$APP_DIR"/*.sh "$APP_DIR"/*.py
}

install_voice_prompts() {
  if [[ "$SKIP_VOICE_PROMPTS" == "1" || -n "$(find "$VOICE_DIR" -maxdepth 1 -name '*.wav' -print -quit 2>/dev/null)" ]]; then
    return
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/boson-ai/higgs-audio.git "$tmp_dir/higgs-audio"
  git -C "$tmp_dir/higgs-audio" sparse-checkout set examples/voice_prompts
  cp -a "$tmp_dir/higgs-audio/examples/voice_prompts/." "$VOICE_DIR/"
  rm -rf "$tmp_dir"
}

print_summary() {
  "$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
import gradio

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("gradio", gradio.__version__)
PY

  echo
  echo "Install complete."
  echo "App dir: $APP_DIR"
  echo "Venv: $VENV_DIR"
  echo "Model: $MODEL_DIR"
  echo "Tokenizer: $TOKENIZER_DIR"
  echo
  echo "Start UI:"
  echo "  $APP_DIR/start_webui.sh"
}

main() {
  install_apt_packages
  install_python_env
  download_models
  install_app_files
  install_voice_prompts
  print_summary
}

main "$@"
