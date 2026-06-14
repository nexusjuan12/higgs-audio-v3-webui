#!/usr/bin/env python3
"""Run local Higgs Audio v3 TTS or voice cloning via the Transformers port."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(APP_DIR / "models" / "higgs-audio-v3-tts-4b-transformers")
DEFAULT_OUTPUT_PATH = str(APP_DIR / "outputs" / "higgs-audio-v3-output.wav")


def read_reference_text(
    ref_audio: str | None,
    explicit_text: str | None,
    text_file: str | None,
) -> str | None:
    if explicit_text:
        return explicit_text.strip()

    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()

    if ref_audio:
        sidecar = Path(ref_audio).with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()

    return None


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


def load_reference_audio(path: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--ref-text-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    dtype = select_dtype(args.dtype, args.device)
    local_files_only = not args.allow_downloads

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        dtype=dtype,
    ).to(args.device).eval()

    reference_audio = None
    reference_sample_rate = None
    reference_text = read_reference_text(args.ref_audio, args.ref_text, args.ref_text_file)
    if args.ref_audio:
        reference_audio, reference_sample_rate = load_reference_audio(args.ref_audio)
        if reference_text is None:
            print(
                "No reference transcript supplied or found beside --ref-audio; "
                "continuing with audio-only conditioning.",
                file=sys.stderr,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        wav = model.generate_speech(
            args.text,
            tokenizer,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            reference_text=reference_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    torchaudio.save(str(out), wav.unsqueeze(0), model.config.sample_rate)
    print(f"wrote {out} ({wav.numel()} samples at {model.config.sample_rate} Hz)")


if __name__ == "__main__":
    main()
