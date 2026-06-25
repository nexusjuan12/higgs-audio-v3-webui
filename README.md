# Higgs Audio v3 Community Web UI

Portable Gradio UI and command-line runner for the community Transformers port of Higgs Audio v3:

- `multimodalart/higgs-audio-v3-tts-4b-transformers`
- `bosonai/higgs-audio-v2-tokenizer`
- `openai/whisper-base` for reference-audio transcription autofill

Higgs Audio v3 is a text-to-speech model intended for expressive speech generation and zero-shot voice cloning. Give it text to synthesize, or provide a short reference audio clip plus its transcript to generate new speech in the reference speaker's voice without training a custom voice model.

The install keeps the environment, models, outputs, logs, and PID file inside the cloned folder by default.

## Install

Clone the repo, then run the installer for your GPU:

```bash
git clone https://github.com/nexusjuan12/higgs-audio-v3-community-webui.git
cd higgs-audio-v3-community-webui

# RTX 5090 / Blackwell
./install_5090.sh

# Tesla P100
./install_p100.sh
```

`./install.sh` is a compatibility wrapper and defaults to the 5090 installer.

Default install paths are relative to the clone:

- Venv: `.venv/`
- Models: `models/`
- Higgs model: `models/higgs-audio-v3-tts-4b-transformers/`
- Audio tokenizer: `models/higgs-audio-v2-tokenizer/`
- Whisper ASR: `models/whisper-base/`
- Voice prompts: `voice_prompts/`
- Outputs: `outputs/`

## Start and Stop

```bash
./start_webui.sh
```

Default local URL:

```text
http://0.0.0.0:7861/
```

On Vast or another hosted GPU box, open the external link mapped to container port `7861`.

Stop the UI:

```bash
./stop_webui.sh
```

## Verify

```bash
./verify_install.sh
```

This writes a short WAV to `outputs/verify-higgs-v3.wav` and prints the sample rate, duration, and peak.

## Transcription

The UI auto-fills `Reference transcript` when reference audio is uploaded or changed. Use the `Transcribe Reference` button to rerun transcription manually. Transcription uses the local `models/whisper-base/` model downloaded during install.

## Voice Prompt Library

Use `Save voice prompt as` to save the current reference audio and transcript as a reusable voice prompt. Saved prompts are stored as paired `.wav` and `.txt` files in `voice_prompts/`, and appear in the `Voice prompt` dropdown on the next refresh/start.

Generated audio is written to `outputs/` and appears in `Output history` for playback or download.

## CLI

Plain TTS:

```bash
.venv/bin/python run_higgs_v3_transformers.py \
  --text "Your text here." \
  --out outputs/output.wav
```

Voice clone:

```bash
.venv/bin/python run_higgs_v3_transformers.py \
  --text "Your cloned voice text here." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio." \
  --out outputs/clone.wav
```

If `--ref-text` is omitted, the runner uses a `.txt` file beside the reference audio when one exists.

Reference audio is downmixed to mono automatically before encoding. This keeps stereo uploads and stereo voice prompt files compatible with the Higgs audio tokenizer.

## Environment Overrides

All scripts accept these environment overrides:

```bash
HIGGS_APP_DIR=/path/to/higgs-audio-v3-webui
HIGGS_VENV=/path/to/higgs-audio-v3-webui/.venv
HIGGS_MODEL_ROOT=/path/to/higgs-audio-v3-webui/models
HIGGS_MODEL_DIR=/path/to/higgs-audio-v3-webui/models/higgs-audio-v3-tts-4b-transformers
HIGGS_TOKENIZER_DIR=/path/to/higgs-audio-v3-webui/models/higgs-audio-v2-tokenizer
HIGGS_ASR_MODEL_DIR=/path/to/higgs-audio-v3-webui/models/whisper-base
HIGGS_VOICE_DIR=/path/to/higgs-audio-v3-webui/voice_prompts
HIGGS_OUTPUT_DIR=/path/to/higgs-audio-v3-webui/outputs
HIGGS_HOST=0.0.0.0
HIGGS_PORT=7861
HIGGS_DTYPE=auto
HIGGS_ASR_DTYPE=auto
```

Rerun helpers:

```bash
SKIP_APT=1 SKIP_PYTHON_ENV=1 SKIP_MODEL_DOWNLOAD=1 ./install_5090.sh
SKIP_APT=1 SKIP_PYTHON_ENV=1 SKIP_MODEL_DOWNLOAD=1 ./install_p100.sh
```

## Notes

The model load may print `audio_head.weight | MISSING`. In this Transformers port, `audio_head.weight` is tied to `audio_embedding.weight` by the model class after loading, so this warning is expected.
