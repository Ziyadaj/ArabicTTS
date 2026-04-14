# ArabicTTS

Fine-tune and serve [Coqui XTTS v2](https://github.com/coqui-ai/TTS) for **Najdi
Arabic** voice cloning, using the [SADA 2022](https://arabicspeech.org/) dataset.

- Data prep: SADA CSVs → segmented wavs + LJSpeech-style manifest
- Fine-tune: Coqui's `GPTTrainer` recipe, tuned for a single **RTX 4080 (16 GB)**
- Serve: Gradio UI + `arabic-tts synth` CLI for one-shot synthesis
- Dockerfile + `docker compose` with NVIDIA GPU passthrough
- Ruff-clean, pytest-tested, CI-gated

## Layout

```
src/arabic_tts/
  cli.py              # `arabic-tts ...` entrypoint (click)
  config.py           # paths + constants
  data/
    text.py           # Arabic text normalization
    sada.py           # SADA CSV → segmented wavs + metadata.csv
    formatter.py      # Coqui dataset formatter for the prepared corpus
  inference/
    model.py          # XTTS v2 loader + synthesize()
    gradio_app.py     # Gradio voice-cloning UI
  training/
    finetune.py       # XTTS v2 GPT fine-tune recipe
scripts/              # convenience runners
tests/                # unit tests (no GPU required)
notebooks/            # historical notebook
```

## Requirements

- **Linux** with an NVIDIA GPU (tested target: RTX 4080, driver ≥ 535, CUDA 12.1)
- **Python 3.10 or 3.11** (Coqui TTS 0.22 is incompatible with 3.12)
- [`uv`](https://github.com/astral-sh/uv) — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker + NVIDIA Container Toolkit (if using the Docker path)
- ~20 GB free disk for XTTS v2 base assets + prepared SADA shards
- SADA 2022 dataset under `data/sada2022/` (you must request access separately)

## Install — bare metal (recommended for training)

```zsh
git clone https://github.com/Ziyadaj/ArabicTTS.git
cd ArabicTTS

uv venv --python 3.11 .venv
source .venv/bin/activate

# Heavy stack: torch cu121 + TTS + gradio
uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -e '.[tts,dev]'

# Sanity check that CUDA is visible.
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Install — Docker (recommended for serving)

```zsh
# Build once.
docker compose build

# Run the Gradio UI on http://localhost:7860
docker compose up

# Run an arbitrary CLI command inside the image (e.g. training).
docker compose run --rm arabic-tts \
  arabic-tts train --dataset data/sada_najdi_prepared --out runs/exp1
```

The compose file mounts `./data`, `./runs`, and `./cache` into the container
and grants access to all NVIDIA GPUs.

## Usage

All subcommands are exposed through a single entry point:

```zsh
arabic-tts --help
```

### 1. Download XTTS v2 base weights

```zsh
arabic-tts download-xtts
```

Drops the five files (`model.pth`, `config.json`, `dvae.pth`, `mel_stats.pth`,
`vocab.json`) into `~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/`.
`arabic-tts train` will also download these on demand into the run directory.

### 2. Prepare the SADA Najdi split

The [SADA 2022](https://arabicspeech.org/) dataset ships long audio files plus
a CSV of utterance segments. Convert to one-wav-per-utterance at 22050 Hz:

```zsh
arabic-tts prepare-sada \
  --csv data/sada2022/train.csv \
  --audio-dir data/sada2022 \
  --out  data/sada_najdi_prepared
```

Writes `data/sada_najdi_prepared/wavs/*.wav` plus `metadata.csv`
(`wav|text|speaker`, LJSpeech-style). The filter keeps segments between 1–11 s
with non-empty normalized Arabic text.

Repeat for `valid.csv` if you want a separate eval split.

### 3. Fine-tune

```zsh
arabic-tts train \
  --dataset data/sada_najdi_prepared \
  --out     runs/xtts_najdi_01 \
  --epochs  10
```

Defaults target a single RTX 4080 (16 GB): `batch_size=2`, `grad_accum=32`
(effective batch = 64), `max_wav_length=11s`. Tune `--batch-size` /
`--grad-accum` if you hit OOM.

Logs go to `runs/xtts_najdi_01/...`; TensorBoard-compatible events are written
alongside checkpoints. Watch training with:

```zsh
tensorboard --logdir runs/
```

### 4. Serve the Gradio UI

```zsh
arabic-tts serve                                   # uses the base XTTS v2
arabic-tts serve --model-dir runs/xtts_najdi_01/best_model  # fine-tuned
```

Open http://localhost:7860. Upload a 5–30 s reference clip, paste Arabic text,
hit Generate.

### 5. One-shot CLI synthesis

```zsh
arabic-tts synth \
  --text "مرحبا، كيف حالك اليوم؟" \
  --speaker-wav samples/reference.wav \
  --out out.wav
```

## Development

```zsh
uv pip install -e '.[dev]'
ruff format src tests scripts
ruff check src tests scripts
pytest -q
```

The `dev` extra is lightweight (no torch / TTS) — enough for lint + unit tests
to run in CI on a stock Ubuntu runner.

## Known gotchas

- **Coqui TTS is archived upstream.** It still installs and works, but is
  pinned in `pyproject.toml` to `TTS==0.22.0` (the last release). Don't
  upgrade `torch`/`transformers` past the pins without testing.
- **Python 3.12 is not supported** by Coqui TTS 0.22 — stick to 3.10 or 3.11.
- **`espeak-ng` must be on PATH** for text-cleaner phoneme fallbacks. The
  Dockerfile installs it; on a host machine run `sudo apt install espeak-ng`.
- **SADA access** is gated; this repo does not ship any dataset files.
- **XTTS license.** The XTTS v2 weights are under the [Coqui Public Model
  License](https://coqui.ai/cpml). Review before shipping anything trained on
  top.

## License

MIT for the code in this repository. See `LICENSE` (TODO). The XTTS v2 weights
downloaded by `arabic-tts download-xtts` are governed by CPML, not MIT.
