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
@click.option(
    "--environments",
    default="Clean",
    show_default=True,
    help="Comma-separated env list (Clean,Car,Music,Noisy). Empty string keeps all.",
)
@click.option(
    "--genders",
    default="male,female",
    show_default=True,
    help="Comma-separated genders (male,female,unknown). Empty string keeps all.",
)
@click.option(
    "--min-utterances", default=5, show_default=True, type=int, help="Drop speakers with fewer."
)
@click.option(
    "--max-utterances",
    default=200,
    show_default=True,
    type=int,
    help="Cap per-speaker utterances. Use -1 to disable.",
)
@click.option(
    "--max-hours",
    default=None,
    type=float,
    help="Optional total-hour budget; randomly subsamples to fit.",
)
def prepare_sada(
    csv_path: Path,
    audio_dir: Path,
    out_dir: Path,
    dialect: str,
    environments: str,
    genders: str,
    min_utterances: int,
    max_utterances: int,
    max_hours: float | None,
) -> None:
    """Segment SADA audio and emit LJSpeech-style metadata."""
    from arabic_tts.data.sada import prepare_split

    env_tuple = tuple(e for e in environments.split(",") if e) or None
    gender_tuple = tuple(g for g in genders.split(",") if g) or None
    cap = None if max_utterances < 0 else max_utterances

    meta = prepare_split(
        csv_path,
        source_audio_dir=audio_dir,
        out_dir=out_dir,
        dialect=dialect,
        environments=env_tuple,
        genders=gender_tuple,
        min_utterances_per_speaker=min_utterances,
        max_utterances_per_speaker=cap,
        max_hours=max_hours,
    )
    click.echo(f"Wrote {meta}")
    ref = out_dir / "reference.txt"
    if ref.is_file():
        click.echo(f"Reference clip: {out_dir / ref.read_text().strip()}")


@main.command("download-sada")
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
    help="Where to put the Kaggle zip and the extracted dataset.",
)
@click.option("--force", is_flag=True, default=False, help="Re-download even if zip is present.")
def download_sada(dest: Path, force: bool) -> None:
    """Pull SADA 2022 from Kaggle and verify the layout."""
    from arabic_tts.data.kaggle_pull import KaggleNotConfiguredError, pull

    try:
        layout = pull(dest, force=force)
    except KaggleNotConfiguredError as e:
        raise click.ClickException(str(e)) from e
    root = layout.pop("__root__")
    click.echo(f"SADA root: {root}")
    for name, path in sorted(layout.items()):
        click.echo(f"  {name}: {path}")


@main.command("inspect-sada")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--dialect", default=None, help="Restrict the report to a single dialect.")
def inspect_sada(csv_path: Path, dialect: str | None) -> None:
    """Report hours by dialect/environment/gender + speaker counts for a split."""
    from arabic_tts.data.inspect import summarize

    report = summarize(csv_path, dialect=dialect)
    click.echo(report.render())


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
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Tiny config (1 epoch, batch 1) to validate the full pipeline end-to-end.",
)
@click.option(
    "--resume",
    "restore_path",
    default=None,
    help="Pass a checkpoint path or `latest` to auto-discover the most recent.",
)
@click.option(
    "--reference-wav",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Speaker reference for the periodic test_sentences synthesis.",
)
def train(
    dataset_path: Path,
    output_path: Path,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    run_name: str,
    smoke: bool,
    restore_path: str | None,
    reference_wav: Path | None,
) -> None:
    """Fine-tune XTTS v2 on the prepared dataset."""
    from arabic_tts.training.finetune import TrainArgs, run

    restore = Path(restore_path) if restore_path else None
    args = TrainArgs(
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
        grad_acumm_steps=grad_accum,
        num_epochs=epochs,
        run_name=run_name,
        smoke=smoke,
        restore_path=restore,
        reference_wav=reference_wav,
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
