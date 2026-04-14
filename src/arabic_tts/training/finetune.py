"""XTTS v2 GPT fine-tune recipe for the prepared SADA Najdi corpus.

Closely mirrors Coqui's reference LJSpeech recipe
(`TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py`) but:
  * uses our SADA Najdi formatter + Arabic language tag,
  * defaults are tuned for a single RTX 4080 (16 GB VRAM),
  * auto-downloads the base XTTS v2 assets if they are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from arabic_tts.data.formatter import sada_najdi_formatter

logger = logging.getLogger(__name__)

# Public URLs for XTTS v2 base assets (used as fine-tune starting point).
# These are the same files Coqui's recipe downloads.
DVAE_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CKPT_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
XTTS_CONFIG_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"


@dataclass
class TrainArgs:
    dataset_path: Path
    output_path: Path
    run_name: str = "xtts_sada_najdi"
    project_name: str = "ArabicTTS"
    language: str = "ar"
    # Tuned for RTX 4080 16GB.
    batch_size: int = 2
    grad_acumm_steps: int = 32
    num_epochs: int = 10
    save_step: int = 1000
    max_wav_length_s: float = 11.0
    max_text_length: int = 200
    min_conditioning_length: int = 66150  # ~3s at 22050
    max_conditioning_length: int = 132300  # ~6s at 22050
    num_loader_workers: int = 8
    meta_file_train: str = "metadata.csv"
    meta_file_val: str | None = None


def _download_if_missing(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlretrieve

    logger.info("Downloading %s → %s", url, dest)
    urlretrieve(url, str(dest))
    return dest


def ensure_base_assets(checkpoints_dir: Path) -> dict[str, Path]:
    """Download XTTS v2 base assets we need for fine-tuning if absent."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "dvae": checkpoints_dir / "dvae.pth",
        "mel_norm": checkpoints_dir / "mel_stats.pth",
        "tokenizer": checkpoints_dir / "vocab.json",
        "xtts_ckpt": checkpoints_dir / "model.pth",
        "xtts_config": checkpoints_dir / "config.json",
    }
    _download_if_missing(DVAE_URL, assets["dvae"])
    _download_if_missing(MEL_NORM_URL, assets["mel_norm"])
    _download_if_missing(TOKENIZER_URL, assets["tokenizer"])
    _download_if_missing(XTTS_CKPT_URL, assets["xtts_ckpt"])
    _download_if_missing(XTTS_CONFIG_URL, assets["xtts_config"])
    return assets


def run(args: TrainArgs) -> Path:
    # Heavy imports are local so this module stays importable for --help / CI.
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
        XttsAudioConfig,
    )

    dataset_path = args.dataset_path.resolve()
    output_path = args.output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_path / "xtts_base"

    assets = ensure_base_assets(checkpoints_dir)

    dataset_config = BaseDatasetConfig(
        formatter="sada_najdi",
        dataset_name="sada_najdi",
        path=str(dataset_path),
        meta_file_train=args.meta_file_train,
        meta_file_val=args.meta_file_val or "",
        language=args.language,
    )

    model_args = GPTArgs(
        max_conditioning_length=args.max_conditioning_length,
        min_conditioning_length=args.min_conditioning_length,
        debug_loading_failures=False,
        max_wav_length=int(args.max_wav_length_s * 22050),
        max_text_length=args.max_text_length,
        mel_norm_file=str(assets["mel_norm"]),
        dvae_checkpoint=str(assets["dvae"]),
        xtts_checkpoint=str(assets["xtts_ckpt"]),
        tokenizer_file=str(assets["tokenizer"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )

    config = GPTTrainerConfig(
        output_path=str(output_path),
        model_args=model_args,
        run_name=args.run_name,
        project_name=args.project_name,
        run_description="Fine-tune XTTS v2 on SADA Najdi Arabic",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=args.num_loader_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=args.save_step,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "مرحبا، كيف حالك اليوم؟",
                "speaker_wav": "",
                "language": args.language,
            }
        ],
        epochs=args.num_epochs,
        grad_clip=0.5,
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.01,
        formatter=sada_najdi_formatter,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_acumm_steps,
        ),
        config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    logger.info("Training done. Run dir: %s", output_path)
    return output_path
