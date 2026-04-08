"""
llm/generator.py — Génération multiple de réponses pour un même prompt
"""

from __future__ import annotations
import time
from typing import List

from llm.provider import BaseLLMProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiResponseGenerator:
    """
    Génère N réponses indépendantes pour un même prompt.
    
    La diversité des réponses est obtenue grâce à la température non nulle
    du provider configuré. Chaque appel est indépendant (pas de mémoire).
    """

    def __init__(self, provider: BaseLLMProvider, delay_between_calls: float = 0.2):
        """
        Args:
            provider: instance du provider LLM
            delay_between_calls: délai en secondes entre chaque appel API (anti rate-limit)
        """
        self.provider = provider
        self.delay = delay_between_calls

    def generate(self, prompt: str, n: int = 3) -> List[str]:
        """
        Génère n réponses pour le prompt donné.

        Args:
            prompt: la question utilisateur
            n: nombre de réponses à générer (min 2)

        Returns:
            Liste de n réponses (strings), la première étant la "réponse principale"

        Raises:
            RuntimeError: si aucune réponse valide n'a pu être générée
        """
        n = max(2, n)
        responses: List[str] = []
        errors: List[str] = []

        logger.info(f"Generating {n} responses for prompt: {prompt[:60]}…")

        for i in range(n):
            try:
                response = self.provider.complete(prompt)
                if response and response.strip():
                    responses.append(response.strip())
                    logger.debug(f"Response {i+1}/{n} generated ({len(response)} chars)")
                else:
                    logger.warning(f"Response {i+1}/{n} was empty, skipping")
            except Exception as e:
                err_msg = str(e)
                errors.append(err_msg)
                logger.error(f"Error generating response {i+1}/{n}: {err_msg}")

            # Pause entre les appels pour éviter le rate limiting
            if i < n - 1:
                time.sleep(self.delay)

        if not responses:
            raise RuntimeError(
                f"Impossible de générer des réponses. Erreurs : {'; '.join(errors)}"
            )

        if len(responses) < n:
            logger.warning(f"Only {len(responses)}/{n} responses were successfully generated")

        return responses

    def generate_with_metadata(self, prompt: str, n: int = 3) -> dict:
        """
        Version enrichie qui retourne les réponses avec des métadonnées.

        Returns:
            dict avec 'responses', 'main_response', 'alt_responses', 'n_requested', 'n_obtained'
        """
        responses = self.generate(prompt, n=n)
        return {
            "responses": responses,
            "main_response": responses[0],
            "alt_responses": responses[1:],
            "n_requested": n,
            "n_obtained": len(responses),
            "prompt": prompt,
        }
