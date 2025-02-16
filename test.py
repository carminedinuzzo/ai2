import os
import time
import torch
import logging
import json
import sys  # Aggiunta l'import di sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="app.log"
)
logger = logging.getLogger(__name__)

# --- Funzioni di Utilità ---

def load_config(file_path):
    try:
        print(f"Tentativo di caricare il file di configurazione da: {file_path}") #Debug
        with open(file_path, 'r', encoding='utf-8') as f:
            config_string = f.read()
            config = json.loads(config_string)
            print(f"File di configurazione caricato correttamente: {config}") #Debug
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {file_path} not found.")
        print(f"Errore: File di configurazione non trovato: {file_path}") #Debug
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}.", exc_info=True)
        print(f"Errore: Errore durante la decodifica JSON dal file: {file_path}: {e}") #Debug
        return {}

if __name__ == "__main__":
    # Load configurations
    config = load_config("parameters.json")
    if not config:
        print("Failed to load config. Exiting...")
        sys.exit(1)

    try:
        # Load Model and Tokenizer
        print("Caricamento modello e tokenizer...")  # Debug print
        print(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        model_name = config.get("MODEL_NAME")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Modello e tokenizer caricati.")  # Debug print
        print(f"Modello: {model}")
        print(f"Tokenizer: {tokenizer}")

        print("Test completato con successo!")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}")  # Print to console as well