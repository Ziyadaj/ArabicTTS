import os
import pandas as pd
import numpy as np
import tempfile
import wave
import librosa
import torch
from torch.utils.data import Dataset
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from trainer import Trainer, TrainerArgs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SADADataset(Dataset):
    def __init__(self, csv_file, audio_dir, config):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['SpeakerDialect'] == 'Najdi'].reset_index(drop=True)
        self.audio_dir = audio_dir
        
        # Default audio settings if not provided in config
        audio_config = config.audio.to_dict() if hasattr(config, 'audio') else {}
        audio_config.setdefault('sample_rate', 16000)
        audio_config.setdefault('frame_length_ms', 50)
        audio_config.setdefault('frame_shift_ms', 12.5)
        audio_config.setdefault('num_mels', 80)
        audio_config.setdefault('fft_size', 1024)
        audio_config.setdefault('win_length', 1024)
        audio_config.setdefault('hop_length', 256)
        
        self.ap = AudioProcessor(**audio_config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['FileName'])
        text = row['ProcessedText']
        
        # Load the full audio file
        full_wav, sr = librosa.load(audio_path, sr=self.ap.sample_rate)
        
        # Calculate start and end samples
        start_sample = int(row['SegmentStart'] * sr)
        end_sample = int(row['SegmentEnd'] * sr)
        
        # Extract the segment
        wav = full_wav[start_sample:end_sample]
        
        # Apply audio processing
        wav = self.ap.trim_silence(wav)
        wav = torch.FloatTensor(wav)

        return {
            "text": text,
            "audio": wav,
            "audio_length": wav.shape[0],
            "speaker_name": row['Speaker'],
            "language_name": "ar"  # Arabic
        }

def setup_base_model():
    logger.info("Setting up base Coqui TTS model")
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, model_path, eval=True)
    return model, config

def prepare_sada_dataset(base_path):
    logger.info("Preparing SADA dataset")
    train_csv = os.path.join(base_path, "train.csv")
    audio_dir = os.path.join(base_path)
    
    model, config = setup_base_model()
    full_data = pd.read_csv(train_csv)
    logger.info(f"Full dataset size: {len(full_data)}")
    
    dataset = SADADataset(train_csv, audio_dir, config)
    logger.info(f"Najdi-only dataset size: {len(dataset)}")
    return dataset, model, config

def fine_tune_model(model, config, dataset, output_path):
    logger.info("Starting fine-tuning")
    # Select 10% of the dataset
    dataset_size = len(dataset)
    subset_size = int(0.1 * dataset_size)
    fine_tune_set = torch.utils.data.Subset(dataset, range(subset_size))
    
    # Set up fine-tuning parameters
    config.batch_size = 2
    config.eval_batch_size = 2
    config.num_loader_workers = 4
    config.num_eval_loader_workers = 4
    config.run_eval = True
    config.test_delay_epochs = -1
    config.epochs = 6
    config.text_cleaner = "phoneme_cleaners"
    config.use_phonemes = True
    config.phoneme_language = "ar"
    config.phoneme_cache_path = os.path.join(output_path, "phoneme_cache")
    config.output_path = output_path

    # Create Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            config_path=config,
            output_path=output_path,
            speakers_file=None,
            language_ids_file=None,
        ),
        config,
        output_path,
        model=model,
        train_samples=fine_tune_set,
        eval_samples=None,
    )

    # Start fine-tuning
    trainer.fit()
    
    logger.info("Fine-tuning completed")
    return trainer.model

def evaluate_model(model, test_set, config):
    logger.info("Evaluating model")
    
    results = []
    for sample in test_set:
        text = sample['text']
        speaker_name = sample['speaker_name']
        
        # Load speaker wav (if required)
        speaker_wav_path = os.path.join(config.output_path, f"{speaker_name}.wav")
        
        if not os.path.isfile(speaker_wav_path):
            logger.warning(f"Speaker wav file {speaker_wav_path} not found.")
            continue

        speaker_wav, _ = librosa.load(speaker_wav_path, sr=config.audio.sample_rate)
        
        # Synthesize speech from text
        generated_audio = model.synthesize(text, speaker_wav=speaker_wav, language="ar")
        
        # For simplicity, we're just logging the generated audio length
        # In a real scenario, you might want to compare this with the original audio
        results.append({"original": text, "generated_length": len(generated_audio)})
        
    return results

if __name__ == "__main__":
    base_path = "data/sada2022"
    
    sada_dataset, base_model, config = prepare_sada_dataset(base_path)
    
    # Evaluate base model
    test_set = torch.utils.data.Subset(sada_dataset, range(len(sada_dataset)-10, len(sada_dataset)))
    base_results = evaluate_model(base_model, test_set, config)
    
    # Fine-tune and evaluate
    fine_tuned_model = fine_tune_model(base_model, config, sada_dataset, "model")
    fine_tuned_results = evaluate_model(fine_tuned_model, test_set, config)
    
    # Compare results
    logger.info("Base Model Results:")
    for result in base_results:
        logger.info(f"Original: {result['original']}")
        logger.info(f"Generated audio length: {result['generated_length']}")
        
    logger.info("Fine-tuned Model Results:")
    for result in fine_tuned_results:
        logger.info(f"Original: {result['original']}")
        logger.info(f"Generated audio length: {result['generated_length']}")