"""
heuristics/detector.py — Détection heuristique de risques dans une réponse LLM

Version améliorée :
- meilleure détection des sujets sensibles
- moins de faux positifs
- prise en compte du type de question
- meilleure gestion des sources
- scoring plus cohérent
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Représente une alerte heuristique déclenchée."""
    name: str
    severity: str          # 'low', 'medium', 'high'
    explanation: str
    weight: float          # 0–1


@dataclass
class HeuristicResult:
    """Résultat complet de l'analyse heuristique."""
    risk_score: float
    risk_level: str        # 'faible', 'moyen', 'élevé'
    alerts: List[Alert] = field(default_factory=list)
    n_alerts: int = 0
    details: Dict[str, object] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Vocabulaire / patterns
# ──────────────────────────────────────────────────────────────

UNCERTAINTY_PATTERNS = [
    r"\bpeut[- ]?être\b",
    r"\bprobablement\b",
    r"\bil semble\b",
    r"\bil paraît\b",
    r"\ben général\b",
    r"\bsouvent\b",
    r"\bparfois\b",
    r"\bdans certains cas\b",
    r"\bselon les sources\b",
    r"\bcertains experts\b",
    r"\bon estime\b",
    r"\bil est possible\b",
    r"\bil se pourrait\b",
    r"\bapproximativement\b",
    r"\benviron\b",
    r"\bil me semble\b",
    r"\bje pense\b",
    r"\bj'imagine\b",
    r"\bil semblerait\b",
    r"\bpotentiellement\b",
    r"\bvraisemblablement\b",
    r"\bapparemment\b",
    r"\bplus ou moins\b",
]

ABSOLUTE_PATTERNS = [
    r"\btoujours\b",
    r"\bjamais\b",
    r"\bcertainement\b",
    r"\babsolument\b",
    r"\bsans aucun doute\b",
    r"\bincontestablement\b",
    r"\bdéfinitivement\b",
    r"\bassurément\b",
    r"\bforcément\b",
    r"\bobligatoirement\b",
    r"\bil est impossible\b",
    r"\bil est certain\b",
    r"\btout le monde\b",
    r"\bpersonne ne\b",
    r"\bdans tous les cas\b",
    r"\bsystématiquement\b",
]

VAGUE_PATTERNS = [
    r"\bbeaucoup\b",
    r"\btrès\b.{0,15}\bimportant\b",
    r"\bvariable\b",
    r"\bdépend\b.{0,30}\bcontexte\b",
    r"\bc'est complexe\b",
    r"\bil faut voir\b",
    r"\bplusieurs facteurs\b",
    r"\bcela varie\b",
    r"\bil n'y a pas de réponse\b",
    r"\bc'est difficile à dire\b",
    r"\bça dépend\b",
    r"\bde nombreux facteurs\b",
    r"\bglobalement\b",
    r"\bplus ou moins\b",
]

FACT_PATTERNS = {
    "percentages": r"\b\d+[,.]?\d*\s*%",
    "years": r"\b(19|20)\d{2}\b",
    "measures": r"\b\d+[,.]?\d*\s*(mg|g|kg|ml|l|cl|km|m|cm|mm|€|\$)\b",
    "versions": r"\b(?:v|version)\s?\d+[.]?\d*\b",
    "currencies": r"\b\d+[,.]?\d*\s*(euros?|dollars?|usd|eur)\b",
    "plain_numbers": r"\b\d+[,.]?\d*\b",
    "proper_names": r"(?<!^)(?<![.!?]\s)(?<!\n)(?:\b[A-ZÉÈÊËÀÂÎÏÔÖÙÛÜÇ][a-zéèêëàâîïôöùûüç]+(?:\s+[A-ZÉÈÊËÀÂÎÏÔÖÙÛÜÇ][a-zéèêëàâîïôöùûüç]+)*)",
}

SOURCE_PATTERNS = [
    r"\bsources?\s*:",
    r"https?://",
    r"\bwww\.",
    r"\bdoi\b",
    r"\bselon\b",
    r"\bd'après\b",
    r"\bsource\b",
    r"\bétude\b",
    r"\brecherche\b",
    r"\brapport\b",
    r"\bpublié\b",
    r"\bjournal\b",
    r"\brevue\b",
    r"\bdocumentation officielle\b",
    r"\bsource officielle\b",
    r"\boms\b",
    r"\binsee\b",
    r"\bansm\b",
    r"\bsanté publique france\b",
    r"\buefa\b",
    r"\bopenai\b",
    r"\banthropic\b",
    r"\bvoir\b\s+\[",
    r"\[\d+\]",
]

FACTUAL_PROMPT_PATTERNS = [
    r"\bqui\b",
    r"\bquand\b",
    r"\bcombien\b",
    r"\bquelle année\b",
    r"\bquel est\b",
    r"\bquelle est\b",
    r"\bquels sont\b",
    r"\bclassement\b",
    r"\bprix\b",
    r"\bcours\b",
    r"\brésultat\b",
    r"\bstatistique\b",
    r"\bdonnée\b",
    r"\bversion\b",
    r"\bcapital(e)?\b",
    r"\bsource\b",
]

OPEN_ENDED_PROMPT_PATTERNS = [
    r"\bexplique\b",
    r"\bcompare\b",
    r"\bavantages\b",
    r"\binconvénients\b",
    r"\bdifférence\b",
    r"\banalyse\b",
    r"\bpourquoi\b",
]

NUANCE_MARKERS = [
    "cependant", "néanmoins", "toutefois", "en revanche", "mais",
    "d'un côté", "d'un autre", "selon le contexte", "à noter que",
    "exception", "sauf si", "dans certains cas", "en pratique",
    "cela dépend", "il faut distinguer", "au contraire",
]

CONTRADICTION_PAIRS = [
    ("toujours", "jamais"),
    ("recommandé", "déconseillé"),
    ("bénéfique", "nuisible"),
    ("sûr", "dangereux"),
    ("légal", "illégal"),
    ("efficace", "inefficace"),
    ("gratuit", "payant"),
    ("vrai", "faux"),
]


SENSITIVE_TOPICS = {
    "santé / médical": {
        "strong": [
            "médicament", "traitement", "diagnostic", "chirurgie", "vaccin",
            "antidépresseur", "antidote", "paracétamol", "ibuprofène",
            "antibiotique", "cancer", "diabète", "grossesse",
            "overdose", "surdosage", "urgence médicale", "ordonnance",
            "posologie", "prescription", "symptômes graves",
        ],
        "weak": [
            "maladie", "symptôme", "allergie", "tension", "fièvre",
            "douleur", "médecin", "dose", "santé", "blessure",
            "soin", "infection",
        ],
    },
    "droit / légal": {
        "strong": [
            "contrat", "tribunal", "procès", "plainte", "jugement",
            "condamnation", "crime", "délit", "avocat", "amende",
            "responsabilité civile", "garde à vue", "licenciement",
            "clause contractuelle", "recours juridique",
        ],
        "weak": [
            "loi", "juridique", "illégal", "légal", "droit", "peine",
            "justice", "réglementation", "litige",
        ],
    },
    "finance / investissement": {
        "strong": [
            "investir", "investissement", "bourse", "action", "actions",
            "crypto", "bitcoin", "ethereum", "rendement", "placement",
            "portefeuille", "risque financier", "trading", "obligation",
            "dividende", "etf", "marché financier", "allocation d'actifs",
            "plus-value", "intérêt composé",
        ],
        "weak": [
            "crédit", "prêt", "impôt", "taxe", "budget", "épargne", "retraite",
            "assurance vie", "financement",
        ],
    },
    "sécurité / cyber": {
        "strong": [
            "arme", "explosif", "piratage", "malware", "phishing",
            "ransomware", "vulnérabilité", "cybersécurité", "attaque informatique",
            "exploit", "payload", "credential stuffing", "ddos",
        ],
        "weak": [
            "hack", "virus informatique", "attaque", "sécurité",
            "mot de passe", "intrusion", "pare-feu",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _count_matches(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def _count_fact_signals(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pattern in FACT_PATTERNS.items():
        matches = re.findall(pattern, text)
        counts[name] = len(matches)
    return counts


def _contains_source(text: str) -> bool:
    return _contains_any_pattern(text, SOURCE_PATTERNS)


def _is_factual_prompt(prompt: str) -> bool:
    return _contains_any_pattern(prompt.lower(), FACTUAL_PROMPT_PATTERNS)


def _is_open_ended_prompt(prompt: str) -> bool:
    return _contains_any_pattern(prompt.lower(), OPEN_ENDED_PROMPT_PATTERNS)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _has_nuance(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in NUANCE_MARKERS)


def _find_contradictions(text: str) -> List[str]:
    t = text.lower()
    contradictions = []

    for a, b in CONTRADICTION_PAIRS:
        if re.search(rf"\b{re.escape(a)}\b", t) and re.search(rf"\b{re.escape(b)}\b", t):
            contradictions.append(f"{a} / {b}")

    for sent in _split_sentences(t):
        if re.search(r"\boui\b", sent) and re.search(r"\bnon\b", sent):
            contradictions.append("oui / non")

    return contradictions


def _detect_sensitive_domain(prompt: str, response: str) -> Tuple[Optional[str], Dict[str, Dict[str, List[str]]]]:
    combined = f"{prompt.lower()} {response.lower()}"
    details: Dict[str, Dict[str, List[str]]] = {}

    for domain, vocab in SENSITIVE_TOPICS.items():
        strong_hits = [kw for kw in vocab["strong"] if re.search(rf"\b{re.escape(kw)}\b", combined)]
        weak_hits = [kw for kw in vocab["weak"] if re.search(rf"\b{re.escape(kw)}\b", combined)]
        details[domain] = {"strong": strong_hits, "weak": weak_hits}

        if len(strong_hits) >= 1 or len(weak_hits) >= 2:
            return domain, details

    return None, details


# ──────────────────────────────────────────────────────────────
# Détecteur principal
# ──────────────────────────────────────────────────────────────

class HeuristicDetector:
    """
    Applique des règles heuristiques sur une réponse LLM.
    """

    SEVERITY_BASE = {
        "low": 12,
        "medium": 22,
        "high": 34,
    }

    MAX_SCORE = 100

    def __init__(self, enabled: Optional[Dict[str, bool]] = None):
        self.enabled = enabled or {
            "uncertainty_vocab": True,
            "vague_language": True,
            "missing_sources": True,
            "unsupported_claims": True,
            "length_check": True,
            "sensitive_topics": True,
            "internal_contradiction": True,
            "absolute_language": True,
            "lack_of_nuance": True,
        }

    def _compute_score(self, alerts: List[Alert]) -> float:
        raw = 0.0
        for alert in alerts:
            base = self.SEVERITY_BASE.get(alert.severity, 15)
            raw += base * alert.weight
        return min(self.MAX_SCORE, round(raw, 1))

    def analyze(self, response: str, prompt: str = "") -> HeuristicResult:
        if not response or not response.strip():
            return HeuristicResult(
                risk_score=55.0,
                risk_level="moyen",
                alerts=[
                    Alert(
                        name="Réponse vide",
                        severity="high",
                        explanation="La réponse est vide ou illisible.",
                        weight=1.0,
                    )
                ],
                n_alerts=1,
                details={"empty_response": True},
            )

        response = _normalize_spaces(response)
        prompt = _normalize_spaces(prompt)

        resp_lower = response.lower()
        prompt_lower = prompt.lower()

        alerts: List[Alert] = []
        details: Dict[str, object] = {}

        fact_counts = _count_fact_signals(response)
        fact_signal_count = (
            fact_counts["percentages"]
            + fact_counts["years"]
            + fact_counts["measures"]
            + fact_counts["versions"]
            + fact_counts["currencies"]
        )

        plain_numbers = fact_counts["plain_numbers"]
        proper_names = fact_counts["proper_names"]
        has_source = _contains_source(response)
        factual_prompt = _is_factual_prompt(prompt)
        open_ended_prompt = _is_open_ended_prompt(prompt)
        word_count = len(response.split())
        sentence_count = len(_split_sentences(response))

        details["fact_counts"] = fact_counts
        details["has_source"] = has_source
        details["factual_prompt"] = factual_prompt
        details["open_ended_prompt"] = open_ended_prompt
        details["word_count"] = word_count
        details["sentence_count"] = sentence_count

        # 1. Vocabulaire d'incertitude
        if self.enabled.get("uncertainty_vocab", True):
            uncertainty_hits = _count_matches(resp_lower, UNCERTAINTY_PATTERNS)
            details["uncertainty_hits"] = len(uncertainty_hits)

            if len(uncertainty_hits) >= 4:
                alerts.append(Alert(
                    name="Vocabulaire d'incertitude",
                    severity="medium",
                    explanation=(
                        f"La réponse contient {len(uncertainty_hits)} marqueurs d'incertitude, "
                        "ce qui peut signaler un manque d'assurance ou de vérification."
                    ),
                    weight=0.8,
                ))
            elif len(uncertainty_hits) >= 2:
                alerts.append(Alert(
                    name="Vocabulaire d'incertitude (léger)",
                    severity="low",
                    explanation=f"Présence de {len(uncertainty_hits)} marqueurs d'incertitude.",
                    weight=0.35,
                ))

        # 2. Langage vague
        if self.enabled.get("vague_language", True):
            vague_hits = _count_matches(resp_lower, VAGUE_PATTERNS)
            details["vague_hits"] = len(vague_hits)

            if len(vague_hits) >= 2:
                alerts.append(Alert(
                    name="Langage vague ou ambigu",
                    severity="medium",
                    explanation="La réponse contient plusieurs formulations vagues et peu actionnables.",
                    weight=0.7,
                ))

        # 3. Absence de sources
        if self.enabled.get("missing_sources", True):
            needs_sources = factual_prompt or fact_signal_count >= 1 or proper_names >= 2
            details["needs_sources"] = needs_sources

            if needs_sources and not has_source:
                severity = "high" if factual_prompt or fact_signal_count >= 2 or proper_names >= 3 else "medium"
                alerts.append(Alert(
                    name="Absence de source",
                    severity=severity,
                    explanation=(
                        "La réponse contient des éléments vérifiables "
                        "(faits, dates, chiffres ou informations précises) sans citer de source."
                    ),
                    weight=0.95 if severity == "high" else 0.7,
                ))

        # 4. Affirmations non justifiées
        if self.enabled.get("unsupported_claims", True):
            if plain_numbers >= 4 and not has_source:
                alerts.append(Alert(
                    name="Affirmations chiffrées sans justification",
                    severity="high",
                    explanation=(
                        f"La réponse contient {plain_numbers} valeurs numériques sans source fiable."
                    ),
                    weight=0.9,
                ))
            elif plain_numbers >= 2 and factual_prompt and not has_source:
                alerts.append(Alert(
                    name="Faits non vérifiés",
                    severity="medium",
                    explanation="La réponse donne plusieurs informations factuelles sans support vérifiable.",
                    weight=0.65,
                ))

        # 5. Longueur suspecte
        if self.enabled.get("length_check", True):
            if word_count < 12:
                alerts.append(Alert(
                    name="Réponse trop courte",
                    severity="medium",
                    explanation=(
                        f"La réponse ne contient que {word_count} mots. "
                        "Elle risque d'être trop brève ou incomplète."
                    ),
                    weight=0.55,
                ))
            elif word_count > 750:
                alerts.append(Alert(
                    name="Réponse anormalement longue",
                    severity="low",
                    explanation=(
                        f"La réponse contient {word_count} mots. "
                        "Une réponse trop longue peut masquer l'information importante."
                    ),
                    weight=0.25,
                ))

        # 6. Sujets sensibles
        if self.enabled.get("sensitive_topics", True):
            domain, sensitive_details = _detect_sensitive_domain(prompt, response)
            details["sensitive_details"] = sensitive_details

            if domain is not None:
                alerts.append(Alert(
                    name=f"Sujet sensible : {domain}",
                    severity="high",
                    explanation=(
                        f"La question ou la réponse concerne le domaine « {domain} ». "
                        "Une réponse incorrecte dans ce domaine peut avoir des conséquences importantes."
                    ),
                    weight=0.95,
                ))

        # 7. Contradiction interne
        if self.enabled.get("internal_contradiction", True):
            contradictions = _find_contradictions(response)
            details["contradictions"] = contradictions

            if contradictions:
                severity = "high" if len(contradictions) >= 2 else "medium"
                alerts.append(Alert(
                    name="Contradiction interne",
                    severity=severity,
                    explanation=(
                        f"La réponse semble contenir des affirmations contradictoires : "
                        f"{', '.join(contradictions[:3])}."
                    ),
                    weight=0.8 if severity == "high" else 0.6,
                ))

        # 8. Formulations absolues
        if self.enabled.get("absolute_language", True):
            absolute_hits = _count_matches(resp_lower, ABSOLUTE_PATTERNS)
            details["absolute_hits"] = len(absolute_hits)

            if len(absolute_hits) >= 2:
                alerts.append(Alert(
                    name="Formulations absolues",
                    severity="medium",
                    explanation=(
                        "La réponse contient plusieurs formulations absolues, "
                        "souvent trop générales ou excessives."
                    ),
                    weight=0.65,
                ))
            elif len(absolute_hits) == 1 and not _has_nuance(response):
                alerts.append(Alert(
                    name="Formulation absolue isolée",
                    severity="low",
                    explanation="La réponse contient une formulation absolue sans nuance explicite.",
                    weight=0.3,
                ))

        # 9. Manque de nuance
        if self.enabled.get("lack_of_nuance", True):
            has_nuance = _has_nuance(response)
            details["has_nuance"] = has_nuance

            if not has_nuance:
                if open_ended_prompt and word_count >= 35:
                    alerts.append(Alert(
                        name="Manque de nuance",
                        severity="medium",
                        explanation=(
                            "La question semble ouverte, mais la réponse ne présente ni nuance, "
                            "ni limites, ni distinction de cas."
                        ),
                        weight=0.6,
                    ))
                elif sentence_count <= 2 and word_count >= 30:
                    alerts.append(Alert(
                        name="Réponse monolithique",
                        severity="low",
                        explanation="La réponse présente une vision unique sans contrepoint ni réserve.",
                        weight=0.35,
                    ))

        risk_score = self._compute_score(alerts)

        if risk_score >= 67:
            risk_level = "élevé"
        elif risk_score >= 34:
            risk_level = "moyen"
        else:
            risk_level = "faible"

        return HeuristicResult(
            risk_score=risk_score,
            risk_level=risk_level,
            alerts=alerts,
            n_alerts=len(alerts),
            details=details,
        )