"""
analysis/coherence.py — Analyse de la cohérence interne entre plusieurs réponses LLM

Méthodes implémentées :
1. Embeddings sémantiques + similarité cosinus (sentence-transformers)
2. Détection de contradictions via NLI (cross-encoder léger, optionnel)
3. Score agrégé de cohérence
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Dataclass résultat ───────────────────────────────────────────────────────

@dataclass
class CoherenceResult:
    """Résultat complet de l'analyse de cohérence interne."""

    score: float                          # 0–100, 100 = parfaitement cohérent
    interpretation: str                   # Texte d'interprétation
    similarity_matrix: Optional[np.ndarray] = None  # Matrice NxN de similarités
    mean_similarity: float = 0.0          # Similarité moyenne entre toutes les paires
    min_similarity: float = 0.0          # Pire paire (le plus divergent)
    contradictions: List[str] = field(default_factory=list)  # Descriptions de contradictions
    embedding_method: str = "cosine"


# ─── Chargement paresseux du modèle d'embedding ───────────────────────────────

_EMBEDDING_MODEL = None

def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Charge le modèle d'embedding une seule fois (lazy loading)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError(
                "Installez sentence-transformers : pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _EMBEDDING_MODEL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─── Analyseur de cohérence ───────────────────────────────────────────────────

class CoherenceAnalyzer:
    """
    Analyse la cohérence interne d'un ensemble de réponses générées pour le même prompt.

    Pipeline :
    1. Encode chaque réponse en vecteur sémantique (sentence-transformers)
    2. Calcule la matrice de similarité cosinus NxN
    3. Agrège les similarités en un score de cohérence
    4. Détecte les paires fortement divergentes (contradictions potentielles)
    5. Analyse lexicale légère pour repérer des inversions sémantiques
    """

    # Seuil en dessous duquel une paire est considérée comme potentiellement contradictoire
    CONTRADICTION_THRESHOLD = 0.65

    # Poids pour le calcul du score final
    # Le score final tient compte à la fois de la moyenne et du minimum
    MEAN_WEIGHT = 0.6
    MIN_WEIGHT = 0.4

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name

    def analyze(self, responses: List[str]) -> CoherenceResult:
        """
        Analyse principale.

        Args:
            responses: liste de réponses pour le même prompt (min 2)

        Returns:
            CoherenceResult avec score, matrice, contradictions
        """
        if len(responses) < 2:
            return CoherenceResult(
                score=50.0,
                interpretation="Analyse impossible : moins de 2 réponses disponibles.",
                mean_similarity=0.5,
                min_similarity=0.5,
            )

        # Filtrage des réponses vides
        valid_responses = [r for r in responses if r and r.strip()]
        if len(valid_responses) < 2:
            return CoherenceResult(
                score=50.0,
                interpretation="Réponses trop courtes ou vides pour l'analyse.",
            )

        try:
            model = _get_embedding_model(self.embedding_model_name)
            embeddings = model.encode(valid_responses, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Embedding failed, using fallback: {e}")
            return self._fallback_analysis(valid_responses)

        n = len(valid_responses)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

        # Extraire les similarités off-diagonal (paires uniques)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(sim_matrix[i][j])

        mean_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.5
        min_sim = float(np.min(pairwise_sims)) if pairwise_sims else 0.5

        # Score pondéré : favorise la stabilité globale ET le cas le plus divergent
        combined_sim = self.MEAN_WEIGHT * mean_sim + self.MIN_WEIGHT * min_sim
        # Transformer en score 0–100
        score = max(0.0, min(100.0, combined_sim * 100))

        # Détecter les contradictions potentielles
        contradictions = self._detect_contradictions(valid_responses, sim_matrix)

        # Bonus/malus lexical léger
        lexical_penalty = self._lexical_contradiction_check(valid_responses)
        score = max(0.0, min(100.0, score - lexical_penalty))

        interpretation = self._interpret(score)

        return CoherenceResult(
            score=round(score, 1),
            interpretation=interpretation,
            similarity_matrix=sim_matrix,
            mean_similarity=round(mean_sim, 3),
            min_similarity=round(min_sim, 3),
            contradictions=contradictions,
            embedding_method="sentence-transformers cosine",
        )

    def _detect_contradictions(
        self, responses: List[str], sim_matrix: np.ndarray
    ) -> List[str]:
        """
        Identifie les paires de réponses avec faible similarité sémantique.
        """
        contradictions = []
        n = len(responses)
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i][j]
                if sim < self.CONTRADICTION_THRESHOLD:
                    # Résumer les deux réponses
                    r1 = responses[i][:60].replace("\n", " ") + "…"
                    r2 = responses[j][:60].replace("\n", " ") + "…"
                    contradictions.append(
                        f"Réponses {i+1} et {j+1} divergent (similarité={sim:.2f}) : "
                        f"« {r1} » ≠ « {r2} »"
                    )
        return contradictions

    def _lexical_contradiction_check(self, responses: List[str]) -> float:
        """
        Pénalité légère basée sur la présence de négations inverses entre réponses.
        Ex : une réponse dit 'oui' et une autre 'non', l'une dit 'toujours' et l'autre 'jamais'.

        Returns:
            pénalité entre 0 et 15
        """
        OPPOSITES = [
            (r"\btoujours\b", r"\bjamais\b"),
            (r"\boui\b", r"\bnon\b"),
            (r"\bdéconseillé\b", r"\brecommandé\b"),
            (r"\bnuisible\b", r"\bbénéfique\b"),
            (r"\bdangereux\b", r"\bsûr\b"),
            (r"\billégal\b", r"\blégal\b"),
        ]
        penalty = 0.0
        for pat_a, pat_b in OPPOSITES:
            has_a = any(re.search(pat_a, r.lower()) for r in responses)
            has_b = any(re.search(pat_b, r.lower()) for r in responses)
            if has_a and has_b:
                penalty += 5.0
        return min(15.0, penalty)

    def _fallback_analysis(self, responses: List[str]) -> CoherenceResult:
        """
        Analyse de secours basée uniquement sur la longueur et le recouvrement lexical.
        Utilisée si sentence-transformers n'est pas disponible.
        """
        def word_overlap(a: str, b: str) -> float:
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / max(len(set_a), len(set_b))

        n = len(responses)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(word_overlap(responses[i], responses[j]))

        mean_sim = float(np.mean(sims)) if sims else 0.5
        score = mean_sim * 100

        return CoherenceResult(
            score=round(score, 1),
            interpretation=self._interpret(score) + " (méthode de secours : recouvrement lexical)",
            mean_similarity=round(mean_sim, 3),
            min_similarity=round(float(min(sims)) if sims else 0.5, 3),
            embedding_method="lexical overlap (fallback)",
        )

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 80:
            return "Très cohérent — les réponses convergent fortement."
        elif score >= 67:
            return "Cohérent — légères variations mais contenu stable."
        elif score >= 50:
            return "Modérément cohérent — divergences notables entre réponses."
        elif score >= 34:
            return "Peu cohérent — le modèle semble instable sur ce sujet."
        else:
            return "Très incohérent — les réponses se contredisent fortement."
