import os
import time
import torch
import psutil
import GPUtil
import optuna
import logging
import json
import random
import requests
import re
import sys
import argparse
import nltk
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge import Rouge
from duckduckgo_search import DDGS
import select
import signal
from typing import Optional, Dict, Any, List, Callable, Tuple  # Aggiunto Tuple
from modules import ModelLoader, WebUtils, PromptBuilder

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
            config = json.load(f)
            print(f"File di configurazione caricato correttamente: {config}") #Debug
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {file_path} not found.")
        print(f"Errore: File di configurazione non trovato: {file_path}") #Debug
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}.", exc_info=True)
        print(f"Errore: Errore durante la decodifica JSON dal file: {file_path}") #Debug
        return {}

class ChatManager:
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module, tokenizer: AutoTokenizer, prompt_builder: PromptBuilder, web_utils: WebUtils):
        print("Inizio inizializzazione ChatManager") #Debug
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.web_utils = web_utils
        self.prompt_builder = prompt_builder # Aggiunta la linea mancante
        self.device = model_loader.device
        self.long_term_memory_file = self.config.get("LONG_TERM_MEMORY_FILE", "long_term_memory.json")
        self.long_term_memory: Dict[str, Any] = self.load_long_term_memory()  # Dictionary to store info
        self.search_cache: Dict[str, str] = {}
        self.bot_name = self.config.get("BOT_NAME", "Noè")
        self.context_separator = self.config.get("CONTEXT_SEPARATOR", "<SEP>")
        self.force_italian = self.config.get("FORCE_ITALIAN", False)
        self.repetition_threshold = self.config.get("REPETITION_THRESHOLD", 3) #Threshold for repeated phrases
        self.use_web_search = self.config.get("USE_WEB_SEARCH", True)
        self.first_turn = True # Track if it's the first turn of the conversation

        #Define available functions:
        self.available_functions: Dict[str, Callable] = {
            "web_search": self.web_search,
            "get_user_name": self.get_user_name
        }
        print("ChatManager inizializzato con successo") #Debug


    def load_long_term_memory(self) -> Dict[str, Any]:
        try:
            print(f"Tentativo di caricare la memoria a lungo termine dal file: {self.long_term_memory_file}") #Debug
            if os.path.exists(self.long_term_memory_file):
                with open(self.long_term_memory_file, "r", encoding="utf-8") as f:
                    try:
                        long_term_memory = json.load(f)
                        logger.info("Memoria a lungo termine caricata dal file.")
                        print(f"Memoria a lungo termine caricata: {long_term_memory}") #Debug
                        return long_term_memory
                    except json.JSONDecodeError:
                        logger.warning("File de memoria a lungo termine corrotto. Inizializzando memoria vuota.")
                        print("Errore: File de memoria a lungo termine corrotto. Inizializzando memoria vuota.") #Debug
                        return {}
            else:
                logger.info("Nessun file de memoria a lungo termine trovato. Inizializzando memoria vuota.")
                print("Nessun file de memoria a lungo termine trovato. Inizializzando memoria vuota.") #Debug
                return {}
        except Exception as e:
            logger.error(f"Errore durante il caricamento dela memoria a lungo termine: {e}", exc_info=True)
            print(f"Errore: Errore durante il caricamento della memoria a lungo termine: {e}") #Debug
            return {}


    def save_long_term_memory(self) -> None:
        """Salva la memoria a lungo termine nel file."""
        try:
            print(f"Tentativo di salvare la memoria a lungo termine nel file: {self.long_term_memory_file}") #Debug
            with open(self.long_term_memory_file, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=4)
            logger.info("Memoria a lungo termine salvata nel file.")
            print("Memoria a lungo termine salvata con successo.") #Debug
        except Exception as e:
            logger.error(f"Errore durante il salvataggio dela memoria a lungo termine: {e}", exc_info=True)
            print(f"Errore: Errore durante il salvataggio della memoria a lungo termine: {e}") #Debug

    def chat(self) -> None:
        """Avvia la modalità chat."""
        logger.info(f"Engaging with {self.bot_name} (type 'exit' to quit).")
        print("Inizio ChatManager.chat()") #Debug Print
        self.first_turn = True # Reset the flag at the beginning of each chat session
        print(f"first_turn è stato impostato a {self.first_turn}")  #Debug print

        while True:
            try:
                user_input = input("You: ").strip()
                print(f"Input utente: {user_input}") #Debug print
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break
            if user_input.lower() == "exit":
                break

            print(f"{self.bot_name}: Sto elaborando la tua richiesta...")
            start_time = time.time()

            prompt = self.prompt_builder.build_prompt(
                user_input, self.format_long_term_memory(), self.available_functions, is_first_turn=self.first_turn
            )
            print(f"Prompt costruito: {prompt}") # Debug print

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            prompt_build_time = time.time() - start_time
            logger.info(f"Tempo per costruire il prompt: {prompt_build_time:.4f} secondi")

            try:
                generation_start_time = time.time()
                response = self.generate_response(input_ids=input_ids, attention_mask=attention_mask)
                generation_time = time.time() - generation_start_time
                logger.info(f"Tempo per generare la risposta: {generation_time:.4f} secondi")
                print(f"Risposta generata: {response}") # Debug print
            except Exception as e:
                logger.error(f"Errore durante la generazione della risposta: {e}", exc_info=True)
                response = "Mi dispiace, si è verificato un errore durante l'elaborazione della richiesta."
                print(f"Errore durante la generazione: {e}") #Debug print

            #Check if the response contains a function call:
            if response.startswith("FUNCTION CALL:"):
                try:
                    function_name, arguments = self.extract_function_call(response)
                    print(f"Funzione chiamata: {function_name}, Argomenti: {arguments}") # Debug print
                    if function_name in self.available_functions:
                        function_to_call = self.available_functions[function_name]
                        function_response = function_to_call(**arguments) #Call the function
                        print(f"Risposta della funzione: {function_response}") # Debug print
                        response = function_response #Override response with function output
                    else:
                        response = f"Error: Function {function_name} not found."
                except Exception as e:
                    response = f"Error during function call: {e}"
                    print(f"Errore durante la chiamata della funzione: {e}") #Debug print


            #Check for repetition:
            if self.is_repeating(response):
                logger.warning(f"Risposta ripetitiva rilevata: {response}")
                response = "Mi dispiace, non sono sicuro di come rispondere a questa domanda."

            print(f"{self.bot_name}: {response} {self.context_separator}")

            self.process_memory_commands(user_input, response)  #Handles all memory related commands

            total_time = time.time() - start_time
            logger.info(f"Tempo totale per l'elaborazione della richiesta: {total_time:.4f} secondi")

            #After the first turn, set to false
            if self.first_turn:
                self.first_turn = False
                print(f"first_turn impostato a False") # Debug print

        print("Fine ChatManager.chat()") #Debug Print


    def extract_function_call(self, response: str) -> tuple[str, Dict[str, Any]]:
        """Extracts the function name and arguments from the response."""
        try:
            function_text = response[len("FUNCTION CALL:"):].strip()
            function_name = function_text.split("(")[0].strip()
            arguments_text = function_text[function_text.find("(") + 1: function_text.find(")")].strip()
            arguments = {}
            if arguments_text:
                for arg in arguments_text.split(","):
                    key, value = arg.split("=")
                    arguments[key.strip()] = value.strip()
            print(f"Funzione estratta: {function_name}, Argomenti: {arguments}") # Debug print
            return function_name, arguments
        except Exception as e:
            logger.error(f"Error extracting function call: {e}", exc_info=True)
            print(f"Errore durante l'estrazione della funzione: {e}") #Debug print
            return None, {}


    def web_search(self, query: str) -> str:
        """Cerca informazioni sul web e restituisce un riassunto."""
        logger.info(f"Esecuzione della funzione web_search con query: {query}")
        print(f"web_search: Esecuzione con query: {query}") #Debug print
        info = self.web_utils.retrieve_information(query)
        if info:
            print(f"web_search: Informazioni trovate: {info}") #Debug print
            return f"Informazioni trovate sul web: {info}"
        else:
            print("web_search: Nessuna informazione trovata sul web.") #Debug print
            return "Nessuna informazione trovata sul web."

    def get_user_name(self) -> str:
        """Restituisce il nome dell'utente dalla memoria, se disponibile."""
        print("get_user_name: Esecuzione") #Debug print
        if "name" in self.long_term_memory:
            print(f"get_user_name: Nome trovato in memoria: {self.long_term_memory['name']}") #Debug print
            return self.long_term_memory["name"]
        else:
            print("get_user_name: Nome non disponibile in memoria.") #Debug print
            return "Nome non disponibile. Per favore, dimmi come ti chiami."


    def is_repeating(self, text: str) -> bool:
        """Checks if the text contains repeating phrases."""
        words = text.split()
        if len(words) < self.repetition_threshold:
            return False

        for i in range(len(words) - self.repetition_threshold):
            phrase = " ".join(words[i:i + self.repetition_threshold])
            if phrase in " ".join(words[i + self.repetition_threshold:]):
                return True
        return False


    def process_memory_commands(self, user_input: str, response: str) -> None:
        """Handles commands related to long-term memory."""
        print("process_memory_commands: Esecuzione") #Debug print

        user_input_lower = user_input.lower()

        if "ricorda che mi chiamo" in user_input_lower:
            name = user_input[len("ricorda che mi chiamo") :].strip()
            self.long_term_memory["name"] = name
            self.save_long_term_memory()
            print(f"{self.bot_name}: Ok, ho memorizzato che ti chiami {name}. {self.context_separator}")

        elif "ti ricordi il mio nome" in user_input_lower: # Keyword trigger
           response = self.get_user_name()
           print(f"{self.bot_name}: {response} {self.context_separator}")

        elif "ricorda che sono" in user_input_lower:
            details = user_input[len("ricorda che sono") :].strip()
            self.long_term_memory["details"] = details
            self.save_long_term_memory()
            print(f"{self.bot_name}: Ho memorizzato le informazioni: {details}. {self.context_separator}")

        elif "aggiorna memoria con" in user_input_lower:
             new_memory = user_input[len("aggiorna memoria con") :].strip()
             self.long_term_memory["misc"] = new_memory #Or append to a list, etc.
             self.save_long_term_memory()
             print(f"{self.bot_name}: Memoria aggiornata con: {new_memory}. {self.context_separator}")


        elif "ti ricordi come mi chiamo" in user_input_lower:
            if "name" in self.long_term_memory:
                response = f"Certo, ti chiami {self.long_term_memory['name']}."
            else:
                response = "Mi dispiace, non mi hai detto come ti chiami."
            print(f"{self.bot_name}: {response} {self.context_separator}")

        elif "quanti anni ho" in user_input_lower:
            if "details" in self.long_term_memory and "anni" in self.long_term_memory["details"]:
                age = self.long_term_memory["details"].split("anni")[0].strip().split()[-1] #crude extraction
                response = f"Hai {age} anni."
            else:
                response = "Non so quanti anni hai."
            print(f"{self.bot_name}: {response} {self.context_separator}")

    def format_long_term_memory(self) -> str:
        """Formatta la memoria a lungo termine per il prompt."""
        print("format_long_term_memory: Esecuzione") #Debug print
        memory_str = ""
        if "name" in self.long_term_memory:
            memory_str += f"Il nome dell'utente è: {self.long_term_memory['name']}. "  #Simplified formatting
        if "details" in self.long_term_memory:
            memory_str += f"Altre informazioni sull'utente: {self.long_term_memory['details']}."
        if "misc" in self.long_term_memory:
            memory_str += f"Altre informazioni utili: {self.long_term_memory['misc']}"
        print(f"format_long_term_memory: Memory_str: {memory_str}") #Debug print
        return memory_str


    def generate_response(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> str:
        """Genera una risposta dal modello."""
        self.model.eval()
        print("generate_response: Esecuzione") #Debug print
        with torch.no_grad():
            generation_kwargs = {
                "temperature": self.config.get("DEFAULT_TEMPERATURE", 0.7),
                "top_p": self.config.get("DEFAULT_TOP_P", 0.9),
                "top_k": self.config.get("DEFAULT_TOP_K", 50),
                "max_new_tokens": self.config.get("DEFAULT_MAX_NEW_TOKENS", 150),
                "repetition_penalty": self.config.get("DEFAULT_REPETITION_PENALTY", 1.1),
                "no_repeat_ngram_size": self.config.get("DEFAULT_NO_REPEAT_NGRAM_SIZE", 2),
                "do_sample": True #required for temperature and top_p
            }

            if input_ids is None:
                 raise ValueError("Devi fornire input_ids.")

            try:
                print("generate_response: Tentativo di generare la risposta...") #Debug print
                output = self.model.generate(
                    input_ids, attention_mask=attention_mask, **generation_kwargs
                )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"generate_response: Risposta generata: {response}") #Debug print


                #Force Italian (more robust attempt):  Translate if necessary
                if self.force_italian:
                    #This is a placeholder;  replace with a proper translation API call if needed.
                    #For example, using Google Translate API or similar.
                    #response = translate(response, target_language='it')
                    pass #No translation implemented for now

                return response

            except Exception as e:
                logger.error(f"Errore durante la generazione: {e}", exc_info=True)
                print(f"generate_response: Errore durante la generazione: {e}") #Debug print
                return "Mi dispiace, ho avuto un problema a rispondere."
