#!/usr/bin/env python3
"""Gradio UI for local Higgs Audio v3 Transformers TTS."""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/root/models/higgs-audio-v3-tts-4b-transformers"
DEFAULT_VOICE_DIR = "/root/higgs-audio/examples/voice_prompts"
DEFAULT_OUTPUT_DIR = "/root/higgs-audio-v3-webui/outputs"


MODEL = None
TOKENIZER = None
DEVICE = None
SAMPLE_RATE = 24000


def select_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device != "cpu" else torch.float32
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


def generate_audio(
    text: str,
    voice_label: str,
    prompt_audio: str | None,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    output_dir: str,
) -> tuple[str, str]:
    if MODEL is None or TOKENIZER is None:
        raise gr.Error("Model is not loaded.")

    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text to synthesize.")

    reference_audio = None
    reference_sample_rate = None
    prompt_audio = prompt_audio or None
    if prompt_audio:
        reference_audio, reference_sample_rate = torchaudio.load(prompt_audio)

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
    status = f"{mode} | {duration:.2f}s audio | {elapsed:.2f}s generation | {out_path}"
    return str(out_path), status


def build_ui(prompts: dict[str, tuple[str, str | None]], output_dir: str) -> gr.Blocks:
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

                output_audio = gr.Audio(label="Output", type="filepath")
                status = gr.Textbox(label="Status", lines=2, interactive=False)

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
        generate.click(
            fn=lambda *args: generate_audio(*args, output_dir),
            inputs=[
                text,
                voice,
                prompt_audio,
                prompt_text,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                seed,
            ],
            outputs=[output_audio, status],
        )
        clear.click(
            fn=lambda: ("", None, "", None, ""),
            outputs=[text, prompt_audio, prompt_text, output_audio, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--voice-dir", default=DEFAULT_VOICE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_model(args.model_path, args.device, args.dtype)
    prompts = discover_voice_prompts(Path(args.voice_dir))
    demo = build_ui(prompts, args.output_dir)
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
