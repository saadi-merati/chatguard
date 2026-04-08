"""
config/settings.py — Chargement de la configuration depuis .env et valeurs par défaut
"""

import os
from dotenv import load_dotenv

load_dotenv()


def load_config() -> dict:
    """
    Charge la configuration depuis les variables d'environnement (.env).
    Retourne un dictionnaire avec toutes les clés de configuration.
    """
    return {
        # Provider
        "default_provider": os.getenv("LLM_PROVIDER", "groq"),
        "default_model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),

        # OpenAI
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),

        # Groq
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),

        # HuggingFace
        "hf_api_token": os.getenv("HUGGINGFACE_API_TOKEN", ""),
        "hf_model_id": os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),

        # Génération
        "n_generations": int(os.getenv("N_GENERATIONS", "3")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),

        # Scoring
        "alpha": float(os.getenv("ALPHA", "0.4")),
        "beta": float(os.getenv("BETA", "0.6")),

        # Seuils de risque
        "threshold_low": int(os.getenv("THRESHOLD_LOW", "33")),
        "threshold_high": int(os.getenv("THRESHOLD_HIGH", "67")),

        # Embeddings
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }


# Instance singleton exportable
CONFIG = load_config()