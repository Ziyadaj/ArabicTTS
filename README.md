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

### 2. Pull SADA 2022 from Kaggle

SADA is gated behind a Kaggle dataset terms click-through. One-time setup:

1. `pip install kaggle` (already in the lite deps).
2. https://www.kaggle.com/settings → **Create New Token** → save `kaggle.json`
   to `~/.kaggle/kaggle.json` (chmod 600).
3. Visit https://www.kaggle.com/datasets/sdaiancai/sada2022 once in a browser
   and accept the dataset terms. The CLI returns 403 until you do this.

Then:

```zsh
arabic-tts download-sada --dest data
```

Pulls the zip into `data/`, unzips into `data/extracted/`, and verifies that
`train.csv`, `valid.csv`, `test.csv`, and the long `.wav` files are present.
Plan for **~50–100 GB** of disk for the raw download and several hundred GB
more if you materialize all dialects as segmented per-utterance wavs (Najdi
alone after filtering is much smaller).

### 3. Inspect what's actually in there

The dataset's per-dialect hour breakdown isn't published, so confirm Najdi is
big enough for your goals before cutting wavs:

```zsh
arabic-tts inspect-sada --csv data/extracted/sada2022/train.csv
arabic-tts inspect-sada --csv data/extracted/sada2022/train.csv --dialect Najdi
```

Reports total hours, per-dialect / per-environment / per-gender hours, and the
top 10 speakers by utterance count.

### 4. Prepare the Najdi split

Convert SADA's long files into one-wav-per-utterance at 22050 Hz with
sensible filters (Clean only, named gender, single speaker, length 1–11 s,
speaker-balanced):

```zsh
arabic-tts prepare-sada \
  --csv       data/extracted/sada2022/train.csv \
  --audio-dir data/extracted/sada2022 \
  --out       data/sada_najdi_prepared
```

Useful knobs:

- `--environments Clean,Music` — keep extra environments (default `Clean`).
- `--genders male,female,unknown` — keep all (default skips unknown).
- `--max-utterances 200` — cap per-speaker so no one voice dominates.
- `--min-utterances 5` — drop speakers with too little data to learn.
- `--max-hours 30` — total-hour budget; randomly subsamples to fit.

Output: `wavs/*.wav`, `metadata.csv` (`wav|text|speaker`), and `reference.txt`
pointing at a clip the trainer will auto-use as the periodic test_sentences
speaker reference. Run again on `valid.csv` for the eval split.

### 5. Fine-tune

```zsh
# Sanity-check the whole pipeline in ~3 minutes before spending hours.
arabic-tts train --dataset data/sada_najdi_prepared --out runs/smoke --smoke

# Real run.
arabic-tts train \
  --dataset data/sada_najdi_prepared \
  --out     runs/xtts_najdi_01 \
  --epochs  30

# Resume from the latest checkpoint.
arabic-tts train --dataset data/sada_najdi_prepared --out runs/xtts_najdi_01 --resume latest
```

Defaults target a single RTX 4080 (16 GB): `batch_size=2`, `grad_accum=32`
(effective batch = 64), `max_wav_length=11s`. Tune `--batch-size` /
`--grad-accum` if you hit OOM. The trainer logs CUDA device info on startup.

Watch training:

```zsh
tensorboard --logdir runs/
```

### 6. Serve the Gradio UI

```zsh
arabic-tts serve                                   # uses the base XTTS v2
arabic-tts serve --model-dir runs/xtts_najdi_01/best_model  # fine-tuned
```

Open http://localhost:7860. Upload a 5–30 s reference clip, paste Arabic text,
hit Generate.

### 7. One-shot CLI synthesis

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

- **Code**: MIT.
- **XTTS v2 weights** (downloaded by `arabic-tts download-xtts`): governed by
  the [Coqui Public Model License](https://coqui.ai/cpml).
- **SADA 2022 dataset** (pulled by `arabic-tts download-sada`): **CC BY-NC-SA
  4.0** — non-commercial, attribution required, share-alike. **Models
  fine-tuned on SADA inherit these terms** and cannot be released
  commercially. Cite Alharbi et al., "SADA: Saudi Audio Dataset for Arabic,"
  IEEE ICASSP 2024.
