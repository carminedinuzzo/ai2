import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bs4 import BeautifulSoup
import requests
import re
from duckduckgo_search import DDGS
from typing import Optional, Dict, Any, List, Callable #Aggiunto Callable
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, config: Dict[str, Any]):
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

    def create_quantization_config(self) -> BitsAndBytesConfig:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def load_model(self) -> torch.nn.Module:
        model_name = self.config.get("MODEL_NAME")
        logger.info(f"Loading model: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=self.create_quantization_config(),
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
        self.config = config
        self.summary_model_name = self.config.get("SUMMARY_MODEL_NAME", "facebook/bart-large-cnn") # Example model
        try:
            self.summary_tokenizer = AutoTokenizer.from_pretrained(self.summary_model_name)
            self.summary_model = AutoModelForCausalLM.from_pretrained(self.summary_model_name)
            self.summary_model.eval()
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}", exc_info=True)
            self.summary_tokenizer = None
            self.summary_model = None


    def browse_web(self, url: str) -> Optional[str]:
        timeout = self.config.get("REQUEST_TIMEOUT", 10)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
            text = re.sub(r"\s+", " ", text).strip()
            return text  # Return full text for summarization
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore nella richiesta HTTP per {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Errore nella navigazione web per {url}: {e}", exc_info=True)
            return None

    def summarize_text(self, text: str, max_length: int = 500, min_length: int = 100) -> str:
        """Summarizes the given text using a pre-trained summarization model."""
        if not self.summary_model or not self.summary_tokenizer:
            logger.warning("Summarization model not loaded. Returning original text.")
            return text[:max_length]  #Basic Truncation fallback

        try:
            inputs = self.summary_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = self.summary_model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            return text[:max_length]  # Truncation fallback


    def retrieve_information(self, query: str) -> Optional[str]: #Removed search_cache
        duckduckgo_max_results = self.config.get("DUCKDUCKGO_MAX_RESULTS", 3)
        max_browsed_length = self.config.get("MAX_BROWSED_LENGTH", 1500) #For fallback

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=duckduckgo_max_results))
                if results:
                    first_result = results[0]
                    title = first_result.get('title', 'No Title')
                    url = first_result.get('href', '')
                    snippet = first_result.get('body', 'No Summary')
                    web_content = self.browse_web(url)

                    if web_content:
                        summary = self.summarize_text(web_content, max_length=max_browsed_length)
                        info = f"{title}\n{snippet}\nRiassunto del contenuto web: {summary}"
                    else:
                        info = f"{title}\n{snippet}\nNessun contenuto web disponibile." # Explicit fallback

                    return info
                else:
                    return None
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}", exc_info=True)
            return None


class PromptBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_name = self.config.get("BOT_NAME")
        self.force_italian = self.config.get("FORCE_ITALIAN", False)
        self.system_prompt = self.config.get("SYSTEM_PROMPT", f"""Sei un assistente AI chiamato {self.bot_name}. Rispondi alle domande in modo conciso e informativo in italiano. Se non conosci la risposta, ammettilo.
        Quando ti viene chiesto di navigare in internet, usa la funzione `web_search(query)` per cercare informazioni e riassumi brevemente le informazioni trovate.
        Quando ti viene chiesto il nome dell'utente, usa la funzione `get_user_name()` per recuperarlo dalla memoria.
        """)


    def build_prompt(
        self, user_input: str, long_term_memory: str, available_functions: Dict[str, Callable], is_first_turn: bool = False
    ) -> str:
        """Costruisce il prompt."""

        prompt = ""
        if is_first_turn:
            prompt += self.system_prompt #Only add system prompt on first turn

        if self.force_italian:
            prompt += " Rispondi sempre in italiano."

        # Aggiungi la memoria a lungo termine
        if long_term_memory:
            prompt += f"\nRicorda: {long_term_memory}"

        #Describe available functions:
        prompt += "\n\nPuoi utilizzare le seguenti funzioni:\n"
        for function_name, function in available_functions.items():
            prompt += f"- `{function_name}(...)`: {function.__doc__}\n"


        prompt += f"\nUtente: {user_input}\n{self.bot_name}:"
        return prompt
