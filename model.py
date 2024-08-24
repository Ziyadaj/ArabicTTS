import os
import pandas as pd
from typing import Dict, List
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Phoneme mapping (based on the Amazon Polly)
PHONEME_MAP: Dict[str, str] = {
    'b': 'b', 'd': 'd', 'dˤ': 'd_?\\', 'f': 'f', 'g': 'g', 'j': 'j',
    'k': 'k', 'l': 'l', 'lˤ': 'I_G', 'm': 'm', 'n': 'n', 'p': 'p',
    'q': 'q', 'r': 'r', 's': 's', 'sˤ': 's_?\\', 't': 't', 'tˤ': 't_?\\',
    'v': 'v', 'w': 'w', 'x': 'x', 'z': 'z', 'ð': 'D', 'ðˤ': 'D_?\\',
    'ħ': 'X\\', 'ŋ': 'N', 'ɣ': 'G', 'ʃ': 'S', 'ʒ': 'Z', 'ʔ': '?',
    'ʕ': '?\\', 'ʤ': 'dZ', 'θ': 'T', 'ɦ': 'h',
    'æ': 'a', 'ɑˤ': 'A_?\\', 'æː': 'a:', 'ɑˤː': 'A_?\\:',
    'a': 'A', 'i': 'i', 'ɪ': 'I', 'iˤ': 'i_?\\', 'iː': 'i:',
    'iˤː': 'i_?:', 'u': 'u', 'ʊ': 'U', 'uˤ': 'u_?\\', 'uː': 'u:',
    'uˤː': 'u_?\\:', 'e': 'e', 'eː': 'e:', 'ɔ': 'O', 'ɔː': 'O:'
}

def text_to_phonemes(text: str) -> str:
    # This is a simplified example. In practice, you'd need a more sophisticated
    # method to convert Arabic text to IPA, then map to X-SAMPA
    phonemes = []
    for char in text:
        if char in PHONEME_MAP:
            phonemes.append(PHONEME_MAP[char])
        else:
            phonemes.append(char)
    return ' '.join(phonemes)


# Load SADA dataset
def load_sada_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df['SpeakerDialect'] == 'Najdi']

# Prepare dataset for XTTS with phoneme conversion
def prepare_xtts_dataset(df: pd.DataFrame, audio_path: str) -> List[Dict[str, str]]:
    dataset = []
    for _, row in df.iterrows():
        phonemes = text_to_phonemes(row['ProcessedText'])
        dataset.append({
            "audio_file": os.path.join(audio_path, row['FileName']),
            "text": row['ProcessedText'],
            "phonemes": phonemes,
            "speaker_name": row['Speaker'],
            "language": "ar"  # Arabic
        })
    return dataset

# Configure and initialize XTTS model
def setup_xtts_model():
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    
    model_path = os.path.join(model_dir, "model.pth")
    config_path = os.path.join(model_dir, "config.json")
    
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, model_path, eval=True)
    return model, config

# Finetune XTTS
def finetune_xtts(model: Xtts, config: XttsConfig, train_data: List[Dict[str, str]], valid_data: List[Dict[str, str]]):
    # Set up finetuning configuration
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60
    config.batch_size = 32
    config.eval_batch_size = 16
    config.num_loader_workers = 4
    config.num_eval_loader_workers = 4
    config.run_eval = True
    config.test_delay_epochs = 5
    config.epochs = 50
    config.use_phonemes = True
    config.phoneme_language = "ar"
    config.phoneme_cache_path = os.path.join(".", "phoneme_cache")
    config.output_path = os.path.join(".", "xtts_najdi_output")

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Start finetuning
    model.fit(train_data, eval_data=valid_data, config=config, ap=ap)


# Main execution
if __name__ == "__main__":
    # Load and prepare data
    train_df = load_sada_data("data/sada2022/train.csv")
    valid_df = load_sada_data("data/sada2022/valid.csv")
    train_data = prepare_xtts_dataset(train_df, "data/sada2022")
    valid_data = prepare_xtts_dataset(valid_df, "data/sada2022")

    # Setup and finetune model
    model, config = setup_xtts_model()
    finetune_xtts(model, config, train_data, valid_data)

    print("Finetuning complete. Model saved in ./xtts_najdi_output")
