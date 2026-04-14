#!/usr/bin/env python
"""Convenience wrapper: download XTTS v2 base assets."""

from arabic_tts.config import default_xtts_dir
from arabic_tts.training.finetune import ensure_base_assets

if __name__ == "__main__":
    assets = ensure_base_assets(default_xtts_dir())
    for k, v in assets.items():
        print(f"{k}: {v}")
