import os
import torch
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
LANGUAGE_CODES = {
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
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hungarian": "hu",
}

def load_model(model_path):
    logger.debug(f"Attempting to load XTTS model from {model_path}")
    if model_path is None:
        raise ValueError("Model path is None")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    config_path = os.path.join(model_path, "config.json")
    model_file = os.path.join(model_path, "model.pth")
    
    logger.debug(f"Checking for config file: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    logger.debug(f"Checking for model file: {model_file}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    logger.debug("Loading config")
    config = XttsConfig()
    config.load_json(config_path)
    logger.debug("Initializing model from config")
    model = Xtts.init_from_config(config)
    logger.debug("Loading model checkpoint")
    model.load_checkpoint(config, checkpoint_path=model_file, checkpoint_dir=model_path)
    if torch.cuda.is_available():
        logger.debug("CUDA is available. Moving model to GPU")
        model.cuda()
    else:
        logger.warning("CUDA is not available. Running on CPU.")
    return model

def generate_speech(text, speaker_wav, language, model):
    logger.debug(f"Generating speech for text: {text}")
    logger.debug(f"Speaker wav: {speaker_wav}")
    logger.debug(f"Language: {language}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    logger.debug("Speech generation complete")
    return out["wav"]

def tts_interface(text, audio_file, language, model):
    logger.debug(f"TTS interface called with text: {text}, audio_file: {audio_file}, language: {language}")
    language_code = LANGUAGE_CODES.get(language, "en")
    generated_speech = generate_speech(text, audio_file, language_code, model)
    return (22050, generated_speech)

def create_demo(model):
    demo = gr.Interface(
        fn=lambda text, audio_file, language: tts_interface(text, audio_file, language, model),
        inputs=[
            gr.Textbox(label="Text to speak", lines=3),
            gr.Audio(label="Speaker reference audio", type="filepath"),
            gr.Dropdown(label="Language", choices=list(LANGUAGE_CODES.keys()), value="English")
        ],
        outputs=gr.Audio(label="Generated Speech"),
        title="XTTS Text-to-Speech Demo",
        description="Enter text, upload a speaker reference audio, and select a language to generate speech.",
    )
    return demo

if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    logger.debug(f"Constructed model path: {model_path}")
    
    try:
        # Load the model
        model = load_model(model_path)
        
        # Create and launch the Gradio interface
        demo = create_demo(model)
        demo.launch()
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Please make sure you have downloaded the XTTS v2 model.")
        logger.info("You can download it using: tts --text 'test' --model_name 'tts_models/multilingual/multi-dataset/xtts_v2'")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error args: {e.args}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        exit(1)