"""Thin wrapper around Coqui XTTS v2 for inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arabic_tts.config import XTTS_SAMPLE_RATE, default_xtts_dir

logger = logging.getLogger(__name__)


@dataclass
class XTTSHandle:
    model: object  # Xtts (kept loose so this module imports without TTS installed)
    config: object
    sample_rate: int = XTTS_SAMPLE_RATE


def load_xtts(model_dir: Path | None = None, *, use_cuda: bool | None = None) -> XTTSHandle:
    """Load XTTS v2 from a local checkpoint directory."""
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = model_dir or default_xtts_dir()
    config_path = model_dir / "config.json"
    model_file = model_dir / "model.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"XTTS config not found: {config_path}")
    if not model_file.is_file():
        raise FileNotFoundError(f"XTTS checkpoint not found: {model_file}")

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)

    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    else:
        logger.warning("CUDA unavailable — running XTTS on CPU (slow).")

    return XTTSHandle(model=model, config=config)


def synthesize(
    handle: XTTSHandle,
    text: str,
    speaker_wav: str | Path,
    *,
    language: str = "ar",
    temperature: float = 0.7,
) -> np.ndarray:
    model = handle.model
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(  # type: ignore[attr-defined]
        audio_path=str(speaker_wav),
        gpt_cond_len=30,
        gpt_cond_chunk_len=4,
        max_ref_length=60,
    )
    out = model.inference(  # type: ignore[attr-defined]
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=temperature,
    )
    return np.asarray(out["wav"], dtype=np.float32)
