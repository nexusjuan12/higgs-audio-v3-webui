# Higgs Audio v3 Web UI

Local Gradio UI and command-line runner for the community Transformers port of Higgs Audio v3:

- `multimodalart/higgs-audio-v3-tts-4b-transformers`
- `bosonai/higgs-audio-v2-tokenizer`

This installer targets the current Vast RTX 5090/CUDA 13 template we are using. It installs a Python venv, downloads the model files, patches the v3 model config to use the local audio tokenizer, and starts the UI on `0.0.0.0:7861`.

## Install

```bash
git clone https://github.com/nexusjuan12/higgs-audio-v3-webui.git /root/higgs-audio-v3-webui
cd /root/higgs-audio-v3-webui
./install.sh
```

The default install paths are:

- App: `/root/higgs-audio-v3-webui`
- Venv: `/root/higgs-audio-v3-env`
- Model: `/root/models/higgs-audio-v3-tts-4b-transformers`
- Audio tokenizer: `/root/models/higgs-audio-v2-tokenizer`
- Outputs: `/root/higgs-audio-v3-webui/outputs`

## Start

```bash
/root/higgs-audio-v3-webui/start_webui.sh
```

Default local URL:

```text
http://0.0.0.0:7861/
```

On Vast, open the external link mapped to container port `7861`.

## Stop

```bash
/root/higgs-audio-v3-webui/stop_webui.sh
```

## Verify

```bash
/root/higgs-audio-v3-webui/verify_install.sh
```

This writes a short WAV to the outputs directory and prints the sample rate, duration, and peak.

## CLI

Plain TTS:

```bash
/root/higgs-audio-v3-env/bin/python /root/higgs-audio-v3-webui/run_higgs_v3_transformers.py \
  --text "Your text here." \
  --out /root/output.wav
```

Voice clone:

```bash
/root/higgs-audio-v3-env/bin/python /root/higgs-audio-v3-webui/run_higgs_v3_transformers.py \
  --text "Your cloned voice text here." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio." \
  --out /root/clone.wav
```

If `--ref-text` is omitted, the runner uses a `.txt` file beside the reference audio when one exists.

## Environment Overrides

All scripts accept these environment overrides:

```bash
HIGGS_APP_DIR=/root/higgs-audio-v3-webui
HIGGS_VENV=/root/higgs-audio-v3-env
HIGGS_MODEL_DIR=/root/models/higgs-audio-v3-tts-4b-transformers
HIGGS_TOKENIZER_DIR=/root/models/higgs-audio-v2-tokenizer
HIGGS_VOICE_DIR=/root/higgs-audio-v3-webui/voice_prompts
HIGGS_OUTPUT_DIR=/root/higgs-audio-v3-webui/outputs
HIGGS_HOST=0.0.0.0
HIGGS_PORT=7861
HIGGS_DTYPE=auto
```

Rerun helpers:

```bash
SKIP_APT=1 SKIP_PYTHON_ENV=1 SKIP_MODEL_DOWNLOAD=1 ./install.sh
```

## Notes

The model load may print `audio_head.weight | MISSING`. In this Transformers port, `audio_head.weight` is tied to `audio_embedding.weight` by the model class after loading, so this warning is expected.
