"""Gradio front-end for XTTS voice cloning with Arabic-first defaults."""

from __future__ import annotations

import logging
from pathlib import Path

from arabic_tts.config import XTTS_SAMPLE_RATE
from arabic_tts.data.text import normalize_arabic
from arabic_tts.inference.model import XTTSHandle, load_xtts, synthesize

logger = logging.getLogger(__name__)

LANGUAGES = {
    "Arabic": "ar",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hungarian": "hu",
}


def build_app(handle: XTTSHandle):
    import gradio as gr

    def tts(text: str, speaker_wav: str, language: str, temperature: float, normalize: bool):
        if not text or not text.strip():
            raise gr.Error("Please enter text to synthesize.")
        if not speaker_wav:
            raise gr.Error("Please upload or record a reference voice clip (5–30s).")
        lang_code = LANGUAGES.get(language, "ar")
        if normalize and lang_code == "ar":
            text = normalize_arabic(text)
        audio = synthesize(handle, text, speaker_wav, language=lang_code, temperature=temperature)
        return (XTTS_SAMPLE_RATE, audio)

    with gr.Blocks(title="Arabic XTTS") as demo:
        gr.Markdown(
            "# Arabic XTTS — voice cloning demo\nUpload a short reference clip, type text, hit generate."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", lines=4, rtl=True, value="مرحبا، كيف حالك اليوم؟")
                speaker = gr.Audio(label="Reference voice (5–30s)", type="filepath")
                language = gr.Dropdown(label="Language", choices=list(LANGUAGES), value="Arabic")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                normalize = gr.Checkbox(
                    value=True, label="Normalize Arabic (strip tashkeel, fix punct)"
                )
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Generated speech", autoplay=False)

        btn.click(tts, inputs=[text, speaker, language, temperature, normalize], outputs=out)

    return demo


def launch(
    model_dir: Path | None = None,
    *,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    handle = load_xtts(model_dir)
    demo = build_app(handle)
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
