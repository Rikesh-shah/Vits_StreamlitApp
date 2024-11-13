import argparse
from transformers import AutoTokenizer, AutoModelForTextToWaveform

def download_model(model_name,token , save_directory):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token =token)
    model = AutoModelForTextToWaveform.from_pretrained(model_name, token=token)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Model and tokenizer saved in {save_directory}")

# Argument parsing
parser = argparse.ArgumentParser(description="Download and save a text-to-waveform model.")
parser.add_argument('lang', type=str, choices=['nl', 'en'], help="Language for the model: 'nl' for Dutch or 'en' for English.")
args = parser.parse_args()

# Set model_name and save_directory based on the provided language
if args.lang == "nl":
    # model_name = "procit001/mms-tts-nld_v4_july_25"
    model_name = "procit006/training_tts_nl_v1.0.6_saskia"
    save_directory_tts = "models/voice_dutch/female_nl_1.0.6"  # Change this to your desired path
else:
    model_name = "procit001/female_english_voice"
    save_directory_tts = "models/voice_english/female_en_april"  # Change this to your desired path

# Download the model
download_model(model_name, token , save_directory_tts)


#python3 helper/model_download.py nl
