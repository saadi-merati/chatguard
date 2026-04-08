"""
llm/provider.py — Couche d'abstraction pour les providers LLM
Supporte : OpenAI, Groq, HuggingFace Inference API, Mock (tests sans API)
"""

from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Interface commune pour tous les providers LLM."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


class OpenAIProvider(BaseLLMProvider):
    """Provider OpenAI via openai>=1.0.0."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model, temperature, max_tokens)
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY manquant dans .env")

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Installez la bibliothèque openai : pip install openai")

    def complete(self, prompt: str) -> str:
        try:
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }

            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Erreur OpenAI : {e}") from e


class GroqProvider(BaseLLMProvider):
    """Provider Groq via le client OpenAI compatible."""

    WEB_SEARCH_MODELS = {"groq/compound", "groq/compound-mini"}

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_web_search: bool = False,
    ):
        super().__init__(model, temperature, max_tokens)
        self.use_web_search = use_web_search

        try:
            from openai import OpenAI

            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY manquant dans .env")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        except ImportError:
            raise ImportError("Installez la bibliothèque openai : pip install openai")

    def _resolve_model(self) -> str:
        if self.use_web_search:
            if self.model in self.WEB_SEARCH_MODELS:
                return self.model
            return "groq/compound-mini"
        return self.model

    def complete(self, prompt: str) -> str:
        try:
            params = {
                "model": self._resolve_model(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }

            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**params)
            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"Erreur Groq : {e}") from e


class HuggingFaceProvider(BaseLLMProvider):
    """Provider HuggingFace via l'Inference API (serverless)."""

    HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("Installez requests : pip install requests")
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        if not self.api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN manquant dans .env")

    def complete(self, prompt: str) -> str:
        url = self.HF_API_URL.format(model=self.model)
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "return_full_text": False,
            },
        }

        if self.max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = self.max_tokens

        try:
            resp = self._requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return str(data)
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise RuntimeError(f"Erreur HuggingFace : {e}") from e


_MOCK_RESPONSES = [
    [
        "Le paracétamol peut être pris jusqu'à 4g par jour pour un adulte, soit 8 comprimés de 500mg. Il faut espacer les prises d'au moins 4 heures.",
        "En général, la dose maximale recommandée de paracétamol est de 3g à 4g par jour selon les sources. Certains médecins conseillent de ne pas dépasser 3g pour les personnes fragiles.",
        "Le paracétamol se prend généralement toutes les 4 à 6 heures. La dose journalière maximale est souvent indiquée autour de 4000mg, mais cela peut varier selon la personne.",
        "Il est possible de prendre du paracétamol plusieurs fois par jour. La limite journalière est probablement autour de 3 à 4 grammes selon les cas.",
    ],
    [
        "Python a été créé par Guido van Rossum et sa première version publique date de 1991. C'est un langage de programmation de haut niveau, interprété et à usage général.",
        "Python est un langage créé par Guido van Rossum, dont la première version est sortie en 1991. Il est connu pour sa lisibilité et sa polyvalence.",
        "Le langage Python a été inventé par Guido van Rossum. La première version publique (0.9.0) a été publiée en février 1991.",
    ],
    [
        "Investir en bourse est généralement rentable sur le long terme. Les actions ont historiquement rapporté environ 7% par an en termes réels.",
        "La bourse peut être risquée et certaines personnes perdent tout leur argent. Il est déconseillé d'investir sans connaissances approfondies.",
        "Investir en bourse est toujours une bonne idée car les marchés montent toujours à long terme. Vous ne pouvez pas perdre si vous êtes patient.",
        "Les investissements boursiers sont très risqués et vous pouvez perdre votre capital. Certains experts recommandent d'éviter la bourse pour les non-professionnels.",
    ],
    [
        "C'est une question complexe qui dépend de nombreux facteurs. Il faudrait analyser la situation en détail pour donner une réponse précise.",
        "Cette question peut avoir plusieurs réponses selon le contexte. En général, on peut dire que cela dépend des circonstances particulières.",
        "Il est difficile de répondre de manière définitive à cette question. Plusieurs perspectives sont possibles et il convient d'être prudent.",
    ],
]


class MockProvider(BaseLLMProvider):
    def __init__(self, model: str = "mock-v1", temperature: float = 0.7, max_tokens: Optional[int] = None):
        super().__init__(model, temperature, max_tokens)
        self._call_count = 0

    def complete(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ["paracétamol", "médicament", "dose", "santé", "médecin"]):
            pool = _MOCK_RESPONSES[0]
        elif any(w in p for w in ["python", "programmation", "langage", "informatique"]):
            pool = _MOCK_RESPONSES[1]
        elif any(w in p for w in ["bourse", "invest", "action", "finance", "argent"]):
            pool = _MOCK_RESPONSES[2]
        else:
            pool = _MOCK_RESPONSES[3]

        time.sleep(0.3)
        idx = self._call_count % len(pool)
        self._call_count += 1
        return pool[idx]


def get_llm_provider(
    provider_name: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    use_web_search: bool = False,
) -> BaseLLMProvider:
    provider_name = provider_name.lower().strip()

    defaults = {
        "openai": "gpt-3.5-turbo",
        "groq": "llama-3.1-8b-instant",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
        "mock": "mock-v1",
    }

    if model is None:
        model = defaults.get(provider_name, "unknown")

    if provider_name == "openai":
        return OpenAIProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider_name == "groq":
        return GroqProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_web_search=use_web_search,
        )
    elif provider_name in ("huggingface", "hf"):
        return HuggingFaceProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider_name == "mock":
        return MockProvider(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(
            f"Provider inconnu : '{provider_name}'. Choisissez parmi : openai, groq, huggingface, mock"
        )