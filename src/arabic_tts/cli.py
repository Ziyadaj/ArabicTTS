"""Command-line entry point: `arabic-tts ...`."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from arabic_tts import __version__
from arabic_tts.config import default_xtts_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
@click.version_option(__version__, prog_name="arabic-tts")
def main() -> None:
    """Fine-tune and serve XTTS v2 for Najdi Arabic."""


@main.command("prepare-sada")
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="SADA split CSV (e.g. train.csv).",
)
@click.option(
    "--audio-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing SADA source wav files.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Where to write wavs/ and metadata.csv.",
)
@click.option("--dialect", default="Najdi", show_default=True)
def prepare_sada(csv_path: Path, audio_dir: Path, out_dir: Path, dialect: str) -> None:
    """Segment SADA audio and emit LJSpeech-style metadata."""
    from arabic_tts.data.sada import prepare_split

    meta = prepare_split(csv_path, source_audio_dir=audio_dir, out_dir=out_dir, dialect=dialect)
    click.echo(f"Wrote {meta}")


@main.command("download-xtts")
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=None,
    help="Override default XTTS download directory.",
)
def download_xtts(dest: Path | None) -> None:
    """Fetch the base XTTS v2 checkpoint + config."""
    from arabic_tts.training.finetune import ensure_base_assets

    dest = dest or default_xtts_dir()
    assets = ensure_base_assets(dest)
    for k, v in assets.items():
        click.echo(f"{k}: {v}")


@main.command("train")
@click.option(
    "--dataset", "dataset_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--out", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--batch-size", default=2, show_default=True, type=int)
@click.option("--grad-accum", default=32, show_default=True, type=int)
@click.option("--epochs", default=10, show_default=True, type=int)
@click.option("--run-name", default="xtts_sada_najdi", show_default=True)
def train(
    dataset_path: Path,
    output_path: Path,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    run_name: str,
) -> None:
    """Fine-tune XTTS v2 on the prepared dataset."""
    from arabic_tts.training.finetune import TrainArgs, run

    args = TrainArgs(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
        grad_acumm_steps=grad_accum,
        num_epochs=epochs,
        run_name=run_name,
    )
    run(args)


@main.command("serve")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to an XTTS checkpoint directory (base or fine-tuned).",
)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=7860, show_default=True, type=int)
@click.option("--share", is_flag=True, default=False)
def serve(model_dir: Path | None, host: str, port: int, share: bool) -> None:
    """Launch the Gradio voice-cloning demo."""
    from arabic_tts.inference.gradio_app import launch

    launch(model_dir, server_name=host, server_port=port, share=share)


@main.command("synth")
@click.option("--text", required=True)
@click.option("--speaker-wav", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--language", default="ar", show_default=True)
def synth(
    text: str, speaker_wav: Path, out_path: Path, model_dir: Path | None, language: str
) -> None:
    """One-shot CLI synthesis to a wav file."""
    import soundfile as sf

    from arabic_tts.config import XTTS_SAMPLE_RATE
    from arabic_tts.data.text import normalize_arabic
    from arabic_tts.inference.model import load_xtts, synthesize

    handle = load_xtts(model_dir)
    if language == "ar":
        text = normalize_arabic(text)
    wav = synthesize(handle, text, speaker_wav, language=language)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, XTTS_SAMPLE_RATE)
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
