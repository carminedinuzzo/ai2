# main.py
import logging
import modules
import torch  # Assicurati che sia importato
import time #Importa la libreria time

# Configura il logging
logging.basicConfig(level=logging.INFO)

# Configurazione
config = {
    "MODEL_NAME": "mistralai/Mistral-7B-Instruct-v0.1",
    "OFFLOAD_FOLDER": "./offload",
    "REQUEST_TIMEOUT": 5,  # Ridotto il timeout per la ricerca web
    "MAX_BROWSED_LENGTH": 500, # Ridotta la lunghezza del testo scaricato dal web
    "DUCKDUCKGO_MAX_RESULTS": 2,  # Ridotto il numero di risultati di ricerca
    "BOT_NAME": "IlMioBot",
    "FORCE_ITALIAN": True,
    "SYSTEM_PROMPT": "Sei un assistente AI utile e conciso. Rispondi alle domande nel modo più accurato possibile. Se non conosci la risposta, ammettilo.",
    "MAX_NEW_TOKENS": 150,  # Riduci da 300 a 150
    "TEMPERATURE": 0.7,       # Valori tipici: 0.7-1.0 (più alto = più creativo)
    "TOP_P": 0.9,            # Valori tipici: 0.7-0.95
    "REPETITION_PENALTY": 1.1, # Penalizza la ripetizione di parole
    "USE_WEB_SEARCH": False,     # Disabilita temporaneamente la ricerca web
    "CACHE_ENABLED": True      # Abilita/Disabilita la cache delle risposte
}

def main():
    try:
        # Carica i moduli
        model_loader = modules.ModelLoader(config)
        tokenizer = model_loader.load_tokenizer()
        model = model_loader.load_model()
        web_utils = modules.WebUtils(config)
        prompt_builder = modules.PromptBuilder(config)
        text_generator = modules.TextGenerator(config, model, tokenizer)  # Inizializza TextGenerator

        # Inizializza la memoria
        short_term_memory = ""  # Memoria a breve termine (conversazione corrente)
        long_term_memory = "So tutto sulla storia d'Italia."  # Memoria a lungo termine
        search_cache = {} # Inizializza la cache di ricerca
        response_cache = {} #Inizializza la cache delle risposte

        while True:  # Loop per interagire continuamente
            user_input = input("Utente: ")
            if user_input.lower() == "esci":
                break

            # 1. Controlla se la risposta è nella cache
            if config.get("CACHE_ENABLED", True) and user_input in response_cache:
                response = response_cache[user_input]
                logging.info("Risposta trovata nella cache.")
            else:
                # Misura il tempo di generazione della risposta
                start_time = time.time()

                # 2. Genera una risposta preliminare SENZA usare il web
                prompt = prompt_builder.build_prompt(user_input, short_term_memory, long_term_memory, "") #No web info inizialmente
                response = text_generator.generate_response(prompt, web_utils, user_input)

                end_time = time.time()
                generation_time = end_time - start_time
                logging.info(f"Tempo di generazione della risposta (senza web): {generation_time:.2f} secondi")

                #DEBUG: Stampa la risposta generata per il debug
                logging.debug(f"Risposta generata inizialmente: {response}")

                # Aggiungi la risposta alla cache
                if config.get("CACHE_ENABLED", True):
                    response_cache[user_input] = response

            print(f"{config['BOT_NAME']}: {response}")

            # Aggiorna la memoria a breve termine
            short_term_memory = f"Utente: {user_input}\n{config['BOT_NAME']}: {response}"

    except Exception as e:
        logging.error(f"Si è verificato un errore: {e}", exc_info=True)

if __name__ == "__main__":
    main()