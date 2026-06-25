#!/usr/bin/env python3
"""Gradio UI for local Higgs Audio v3 Transformers TTS."""

from __future__ import annotations

import argparse
import re
import shutil
import time
import uuid
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(APP_DIR / "models" / "higgs-audio-v3-tts-4b-transformers")
DEFAULT_ASR_MODEL_PATH = str(APP_DIR / "models" / "whisper-base")
DEFAULT_VOICE_DIR = str(APP_DIR / "voice_prompts")
DEFAULT_OUTPUT_DIR = str(APP_DIR / "outputs")


MODEL = None
TOKENIZER = None
ASR_MODEL = None
ASR_PROCESSOR = None
ASR_DEVICE = None
DEVICE = None
SAMPLE_RATE = 24000
ASR_SAMPLE_RATE = 16000


def select_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        if device == "cpu":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def discover_voice_prompts(voice_dir: Path) -> dict[str, tuple[str, str | None]]:
    prompts: dict[str, tuple[str, str | None]] = {"None": ("", None)}
    if not voice_dir.exists():
        return prompts

    for wav in sorted(voice_dir.glob("*.wav")):
        label = wav.stem.replace("_", " ").title()
        text_path = wav.with_suffix(".txt")
        transcript = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else None
        prompts[label] = (str(wav), transcript)
    return prompts


def voice_prompt_choices(prompts: dict[str, tuple[str, str | None]]) -> list[str]:
    return list(prompts.keys())


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", name.strip())
    stem = re.sub(r"\s+", "_", stem).strip("._- ")
    if not stem:
        raise gr.Error("Enter a voice prompt save name.")
    return stem


def save_voice_prompt(
    prompt_audio: str | None,
    prompt_text: str,
    save_name: str,
    voice_dir: str,
    prompts: dict[str, tuple[str, str | None]],
) -> str | None:
    save_name = (save_name or "").strip()
    if not save_name:
        return None
    if not prompt_audio:
        raise gr.Error("Choose reference audio before saving a voice prompt.")

    stem = safe_stem(save_name)
    out_dir = Path(voice_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f"{stem}.wav"
    text_path = out_dir / f"{stem}.txt"

    if Path(prompt_audio).resolve() != audio_path.resolve():
        shutil.copy2(prompt_audio, audio_path)
    text_path.write_text((prompt_text or "").strip() + "\n", encoding="utf-8")

    label = audio_path.stem.replace("_", " ").title()
    prompts[label] = (str(audio_path), (prompt_text or "").strip())
    return label


def output_history_choices(output_dir: str, limit: int = 100) -> list[tuple[str, str]]:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return []

    files = sorted(out_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    choices = []
    for wav in files[:limit]:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wav.stat().st_mtime))
        choices.append((f"{timestamp}  {wav.name}", str(wav)))
    return choices


def selected_history_audio(path: str | None) -> str | None:
    return path or None


def load_model(model_path: str, device: str, dtype_name: str) -> None:
    global MODEL, TOKENIZER, DEVICE, SAMPLE_RATE

    DEVICE = device
    dtype = select_dtype(dtype_name, device)
    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
    ).to(device).eval()
    SAMPLE_RATE = int(MODEL.config.sample_rate)


def load_asr_model(model_path: str, device: str, dtype_name: str) -> None:
    global ASR_MODEL, ASR_PROCESSOR, ASR_DEVICE

    if ASR_MODEL is not None and ASR_PROCESSOR is not None and ASR_DEVICE == device:
        return

    ASR_DEVICE = device
    dtype = torch.float32 if device == "cpu" else select_dtype(dtype_name, device)
    ASR_PROCESSOR = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    ASR_MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
    ).to(device).eval()


def selected_voice(
    voice_label: str,
    prompts: dict[str, tuple[str, str | None]],
) -> tuple[str | None, str]:
    audio_path, transcript = prompts.get(voice_label, ("", None))
    return audio_path or None, transcript or ""


def coerce_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if value <= 0 else value


def coerce_optional_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def load_reference_audio(path: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), sample_rate


def transcribe_reference_audio(
    prompt_audio: str | None,
    asr_model_path: str,
    asr_device: str,
    asr_dtype: str,
) -> tuple[str, str]:
    if not prompt_audio:
        return "", "Reference audio cleared."

    try:
        load_asr_model(asr_model_path, asr_device, asr_dtype)
        waveform, sample_rate = load_reference_audio(prompt_audio)
        if sample_rate != ASR_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, ASR_SAMPLE_RATE)

        audio = waveform.squeeze(0).float().cpu().numpy()
        inputs = ASR_PROCESSOR(
            audio,
            sampling_rate=ASR_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(asr_device)
        if asr_device != "cpu":
            input_features = input_features.to(next(ASR_MODEL.parameters()).dtype)

        start = time.time()
        with torch.inference_mode():
            predicted_ids = ASR_MODEL.generate(input_features, max_new_tokens=128)
        transcript = ASR_PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        raise gr.Error(f"Reference transcription failed: {exc}") from exc

    if not transcript:
        raise gr.Error("Reference transcription returned empty text.")

    elapsed = time.time() - start
    return transcript, f"Transcribed reference audio | {elapsed:.2f}s"


def generate_audio(
    text: str,
    voice_label: str,
    prompt_audio: str | None,
    prompt_text: str,
    save_voice_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    output_dir: str,
    voice_dir: str,
    prompts: dict[str, tuple[str, str | None]],
) -> tuple[str, str, gr.Dropdown, gr.Dropdown, str]:
    if MODEL is None or TOKENIZER is None:
        raise gr.Error("Model is not loaded.")

    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text to synthesize.")

    reference_audio = None
    reference_sample_rate = None
    prompt_audio = prompt_audio or None
    if prompt_audio:
        reference_audio, reference_sample_rate = load_reference_audio(prompt_audio)

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    start = time.time()
    with torch.inference_mode():
        wav = MODEL.generate_speech(
            text,
            TOKENIZER,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            reference_text=(prompt_text or "").strip() or None,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=coerce_optional_float(top_p),
            top_k=coerce_optional_int(top_k),
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"higgs-v3-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.wav"
    torchaudio.save(str(out_path), wav.unsqueeze(0), SAMPLE_RATE)

    duration = wav.numel() / SAMPLE_RATE if SAMPLE_RATE else 0
    elapsed = time.time() - start
    mode = "Voice clone" if prompt_audio else "TTS"
    if voice_label and voice_label != "None":
        mode = f"{mode} ({voice_label})"

    saved_label = save_voice_prompt(prompt_audio, prompt_text, save_voice_name, voice_dir, prompts)
    status = f"{mode} | {duration:.2f}s audio | {elapsed:.2f}s generation | {out_path}"
    if saved_label:
        status = f"{status} | saved voice prompt: {saved_label}"

    return (
        str(out_path),
        status,
        gr.Dropdown(choices=voice_prompt_choices(prompts), value=saved_label or voice_label),
        gr.Dropdown(choices=output_history_choices(output_dir), value=str(out_path)),
        str(out_path),
    )


def build_ui(
    prompts: dict[str, tuple[str, str | None]],
    voice_dir: str,
    output_dir: str,
    asr_model_path: str,
    asr_device: str,
    asr_dtype: str,
) -> gr.Blocks:
    with gr.Blocks(title="Higgs Audio v3", fill_height=True) as demo:
        gr.Markdown("# Higgs Audio v3")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=420):
                text = gr.Textbox(
                    label="Text",
                    lines=7,
                    value="This is a local Higgs Audio v3 test.",
                )
                with gr.Row():
                    generate = gr.Button("Generate", variant="primary")
                    clear = gr.Button("Clear")

                output_audio = gr.Audio(label="Output", type="filepath", autoplay=True)
                status = gr.Textbox(label="Status", lines=2, interactive=False)

                with gr.Accordion("Output history", open=True):
                    output_history = gr.Dropdown(
                        choices=output_history_choices(output_dir),
                        value=None,
                        label="Generated files",
                    )
                    with gr.Row():
                        refresh_history = gr.Button("Refresh History")
                    history_audio = gr.Audio(
                        label="Selected output",
                        type="filepath",
                        buttons=["download"],
                    )

            with gr.Column(scale=4, min_width=360):
                voice = gr.Dropdown(
                    choices=list(prompts.keys()),
                    value="None",
                    label="Voice prompt",
                )
                prompt_audio = gr.Audio(
                    label="Reference audio",
                    type="filepath",
                    sources=["upload"],
                )
                prompt_text = gr.Textbox(label="Reference transcript", lines=4)
                save_voice_name = gr.Textbox(
                    label="Save voice prompt as",
                    placeholder="Optional; saved when Generate succeeds",
                )
                transcribe = gr.Button("Transcribe Reference")

                with gr.Accordion("Generation", open=True):
                    max_new_tokens = gr.Slider(
                        128,
                        4096,
                        value=2048,
                        step=64,
                        label="Max new tokens",
                    )
                    temperature = gr.Slider(
                        0.1,
                        1.5,
                        value=1.0,
                        step=0.05,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.01,
                        label="Top p",
                    )
                    top_k = gr.Slider(
                        0,
                        200,
                        value=0,
                        step=1,
                        label="Top k",
                    )
                    seed = gr.Number(value=-1, precision=0, label="Seed")

        voice.change(
            fn=lambda voice_label: selected_voice(voice_label, prompts),
            inputs=[voice],
            outputs=[prompt_audio, prompt_text],
        )
        refresh_history.click(
            fn=lambda: gr.Dropdown(choices=output_history_choices(output_dir), value=None),
            outputs=[output_history],
        )
        output_history.change(
            fn=selected_history_audio,
            inputs=[output_history],
            outputs=[history_audio],
        )
        prompt_audio.change(
            fn=lambda audio: transcribe_reference_audio(
                audio,
                asr_model_path,
                asr_device,
                asr_dtype,
            ),
            inputs=[prompt_audio],
            outputs=[prompt_text, status],
        )
        transcribe.click(
            fn=lambda audio: transcribe_reference_audio(
                audio,
                asr_model_path,
                asr_device,
                asr_dtype,
            ),
            inputs=[prompt_audio],
            outputs=[prompt_text, status],
        )
        generate.click(
            fn=lambda *args: generate_audio(*args, output_dir, voice_dir, prompts),
            inputs=[
                text,
                voice,
                prompt_audio,
                prompt_text,
                save_voice_name,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                seed,
            ],
            outputs=[output_audio, status, voice, output_history, history_audio],
        )
        clear.click(
            fn=lambda: ("", None, "", "", None, ""),
            outputs=[text, prompt_audio, prompt_text, save_voice_name, output_audio, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--asr-model-path", default=DEFAULT_ASR_MODEL_PATH)
    parser.add_argument("--voice-dir", default=DEFAULT_VOICE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asr-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--asr-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_model(args.model_path, args.device, args.dtype)
    prompts = discover_voice_prompts(Path(args.voice_dir))
    demo = build_ui(
        prompts,
        args.voice_dir,
        args.output_dir,
        args.asr_model_path,
        args.asr_device,
        args.asr_dtype,
    )
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
