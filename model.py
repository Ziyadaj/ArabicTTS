import os
import pandas as pd
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Load SADA dataset
def load_sada_data(csv_path):
    df = pd.read_csv(csv_path)
    return df[df["SpeakerDialect"] == "Najdi"]


# Prepare dataset for XTTS
def prepare_xtts_dataset(df, audio_path):
    dataset = []
    for _, row in df.iterrows():
        dataset.append(
            {
                "audio_file": os.path.join(audio_path, row["FileName"]),
                "text": row["ProcessedText"],
                "speaker_name": row["Speaker"],
                "language": "ar",  # Arabic
            }
        )
    return dataset


# Configure and initialize XTTS model
def setup_xtts_model():
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, model_path, eval=True)
    return model, config


# Finetune XTTS
def finetune_xtts(model, config, train_data, valid_data):
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
    config.text_cleaner = "phoneme_cleaners"
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
    train_df = load_sada_data("train.csv")
    valid_df = load_sada_data("valid.csv")
    train_data = prepare_xtts_dataset(train_df, "path/to/audio/files")
    valid_data = prepare_xtts_dataset(valid_df, "path/to/audio/files")

    # Setup and finetune model
    model, config = setup_xtts_model()
    finetune_xtts(model, config, train_data, valid_data)

    print("Finetuning complete. Model saved in ./xtts_najdi_output")
