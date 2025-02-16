import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
import requests
import re
from duckduckgo_search import DDGS
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config: Dict[str, Any]):
        config = dict(config)  # Forza il tipo dict (sicurezza)
        self.config = config
        self.device = self.get_device()
        self.offload_folder = self.config.get("OFFLOAD_FOLDER", "./offload")

    def get_device(self) -> torch.device:
        logger.info(f"CUDA Disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Nome GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"Memoria GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        return device

    def load_model(self) -> torch.nn.Module:
        model_name = self.config.get("MODEL_NAME")
        logger.info(f"Loading model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder=self.offload_folder,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                trust_remote_code=True,
            )
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def load_tokenizer(self) -> AutoTokenizer:
        model_name = self.config.get("MODEL_NAME")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}", exc_info=True)
            raise
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        logger.info(f"Tokenizer loaded: {tokenizer.name_or_path}")
        return tokenizer


class WebUtils:
    def __init__(self, config: Dict[str, Any]):
        config = dict(config)  # Forza il tipo dict (sicurezza)
        self.config = config

    def browse_web(self, url: str) -> Optional[str]:
        timeout = self.config.get("REQUEST_TIMEOUT", 10)
        max_length = self.config.get("MAX_BROWSED_LENGTH", 1500)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_length]
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore nella richiesta HTTP per {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Errore nella navigazione web per {url}: {e}", exc_info=True)
            return None

    def retrieve_information(self, query: str, search_cache: Dict[str, str]) -> Optional[str]:
        duckduckgo_max_results = self.config.get("DUCKDUCKGO_MAX_RESULTS", 3)
        if query in search_cache:
            logger.info(f"Using cached search result for query: {query}")
            return search_cache[query]

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=duckduckgo_max_results))
                logger.debug(f"DuckDuckGo results: {results}")

                if results:
                    first_result = results[0]
                    logger.debug(f"First result: {first_result}, type: {type(first_result)}")

                    if isinstance(first_result, dict):  # Controllo del tipo ESSENZIALE
                        title = first_result.get('title', 'No Title')
                        url = first_result.get('href', '')
                        snippet = first_result.get('body', 'No Summary')
                        web_content = self.browse_web(url)
                        info = f"{title}\n{snippet}\n{web_content or ''}"
                        search_cache[query] = info
                        return info
                    else:
                        logger.warning(f"Unexpected result type from DuckDuckGo: {type(first_result)}")
                        return None
                else:
                    return None
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}", exc_info=True)
            return None


class PromptBuilder:
    def __init__(self, config: Dict[str, Any]):
        config = dict(config)  # Forza il tipo dict (sicurezza)
        self.config = config
        self.bot_name = self.config.get("BOT_NAME")
        self.force_italian = self.config.get("FORCE_ITALIAN", False)
        self.system_prompt = self.config.get("SYSTEM_PROMPT", "Sei un assistente AI utile.")  # System prompt

    def build_prompt(
        self, user_input: str, short_term_memory: str, long_term_memory: str, web_info: str = ""
    ) -> str:
        """Costruisce il prompt per il modello."""

        prompt = self.system_prompt  # Usa il system prompt configurabile

        if self.force_italian:
            prompt += " Rispondi sempre in italiano."

        if long_term_memory:
            prompt += f"\nRicorda: {long_term_memory}"

        if short_term_memory:
            prompt += f"\nUltima interazione: {short_term_memory}"

        if web_info:
            prompt += f"\nInformazioni dal web: {web_info}"  # Aggiungi le informazioni dal web

        prompt += f"\nUtente: {user_input}\n{self.bot_name}:"
        return prompt

class TextGenerator:
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module, tokenizer: AutoTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Recupera il device qui
        self.max_new_tokens = self.config.get("MAX_NEW_TOKENS", 256)  # Imposta un valore di default
        self.temperature = self.config.get("TEMPERATURE", 0.7)
        self.top_p = self.config.get("TOP_P", 0.9)
        self.repetition_penalty = self.config.get("REPETITION_PENALTY", 1.1)
        self.use_web_search = self.config.get("USE_WEB_SEARCH", True) #Flag per attivare o disattivare la ricerca web



    def generate_response(self, prompt: str, web_utils: "WebUtils", user_input: str) -> str:
        """Genera una risposta dal modello, potenzialmente usando la ricerca web."""
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Pulisci la risposta rimuovendo il prompt originale
            user_input_index = prompt.find("Utente:")
            if user_input_index != -1:
                response = response[user_input_index + len("Utente:") + len(prompt.split("Utente:")[1]):].strip()
            else:
                response = response.replace(prompt.split("Utente:")[0], "").strip()

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "Si è verificato un errore durante la generazione della risposta."