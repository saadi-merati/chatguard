"""
scoring/aggregator.py — Fusion pondérée des scores de cohérence et heuristique

Formule :
    final_risk = alpha * incoherence_risk + beta * heuristic_risk
    incoherence_risk = 100 - coherence_score

Seuils :
    0–33  → risque faible
    34–66 → risque moyen
    67–100 → risque élevé
"""

from __future__ import annotations
from dataclasses import dataclass

from analysis.coherence import CoherenceResult
from heuristics.detector import HeuristicResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FinalResult:
    """Résultat agrégé final."""
    final_score: float       # 0–100
    risk_level: str          # 'faible', 'moyen', 'élevé'
    justification: str       # Explication textuelle
    coherence_contribution: float   # Part de l'incohérence dans le score final
    heuristic_contribution: float   # Part heuristique dans le score final
    alpha: float
    beta: float


class ScoreAggregator:
    """
    Agrège les scores de cohérence interne et d'heuristiques en un score global.

    Paramètres configurables :
        alpha : poids de l'incohérence (défaut 0.4)
        beta  : poids du risque heuristique (défaut 0.6)

    Note : alpha + beta n'a pas besoin de faire 1.0 — si ce n'est pas le cas,
    le score final est ramené dans [0, 100] par un clamp simple.
    Si alpha + beta = 1.0, la formule est une moyenne pondérée normalisée.
    """

    THRESHOLD_LOW = 33
    THRESHOLD_HIGH = 67

    def __init__(self, alpha: float = 0.4, beta: float = 0.6):
        self.alpha = alpha
        self.beta = beta

    def compute(
        self,
        coherence_result: CoherenceResult,
        heuristic_result: HeuristicResult,
    ) -> FinalResult:
        """
        Calcule le score final de risque.

        Args:
            coherence_result: résultat de l'analyse de cohérence
            heuristic_result: résultat de l'analyse heuristique

        Returns:
            FinalResult avec score, niveau et justification
        """
        incoherence_risk = 100.0 - coherence_result.score
        heuristic_risk = heuristic_result.risk_score

        coherence_contrib = self.alpha * incoherence_risk
        heuristic_contrib = self.beta * heuristic_risk

        raw_score = coherence_contrib + heuristic_contrib
        final_score = max(0.0, min(100.0, raw_score))

        risk_level = self._classify(final_score)
        justification = self._justify(
            final_score, risk_level, coherence_result, heuristic_result,
            coherence_contrib, heuristic_contrib,
        )

        logger.info(
            f"Score aggregation: incoherence={incoherence_risk:.1f} "
            f"heuristic={heuristic_risk:.1f} "
            f"final={final_score:.1f} level={risk_level}"
        )

        return FinalResult(
            final_score=round(final_score, 1),
            risk_level=risk_level,
            justification=justification,
            coherence_contribution=round(coherence_contrib, 1),
            heuristic_contribution=round(heuristic_contrib, 1),
            alpha=self.alpha,
            beta=self.beta,
        )

    def _classify(self, score: float) -> str:
        if score >= self.THRESHOLD_HIGH:
            return "élevé"
        elif score >= self.THRESHOLD_LOW:
            return "moyen"
        else:
            return "faible"

    def _justify(
        self,
        score: float,
        level: str,
        coh: CoherenceResult,
        heur: HeuristicResult,
        coh_contrib: float,
        heur_contrib: float,
    ) -> str:
        parts = []

        # Contribution cohérence
        if coh.score < 50:
            parts.append(
                f"Le modèle a produit des réponses **incohérentes** (cohérence : {coh.score:.0f}/100), "
                f"contribuant {coh_contrib:.1f} points au score de risque."
            )
        elif coh.score < 70:
            parts.append(
                f"La cohérence est **modérée** ({coh.score:.0f}/100), "
                f"contribution de {coh_contrib:.1f} points."
            )
        else:
            parts.append(
                f"La cohérence est **bonne** ({coh.score:.0f}/100), "
                f"contribution faible de {coh_contrib:.1f} points."
            )

        # Contribution heuristiques
        if heur.n_alerts == 0:
            parts.append("Aucune alerte heuristique détectée.")
        else:
            severity_summary = {}
            for a in heur.alerts:
                severity_summary[a.severity] = severity_summary.get(a.severity, 0) + 1
            detail = ", ".join(f"{v} {k}" for k, v in severity_summary.items())
            parts.append(
                f"{heur.n_alerts} alerte(s) heuristique(s) détectée(s) ({detail}), "
                f"contribuant {heur_contrib:.1f} points."
            )

        # Verdict final
        if level == "élevé":
            parts.append(
                "⛔ **Cette réponse présente un risque élevé.** "
                "Elle nécessite une vérification auprès d'une source fiable avant toute utilisation."
            )
        elif level == "moyen":
            parts.append(
                "⚠️ **Risque modéré.** "
                "Quelques signaux d'alerte ont été détectés. "
                "Il est conseillé de croiser cette information avec d'autres sources."
            )
        else:
            parts.append(
                "✅ **Risque faible.** "
                "La réponse semble relativement fiable selon les critères analysés, "
                "mais aucun système automatique n'est infaillible."
            )

        return " ".join(parts)
