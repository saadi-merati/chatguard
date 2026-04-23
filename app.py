from pathlib import Path
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from analysis.coherence import CoherenceAnalyzer
from config.settings import load_config
from heuristics.detector import HeuristicDetector
from llm.generator import MultiResponseGenerator
from llm.provider import get_llm_provider
from scoring.aggregator import ScoreAggregator
from utils.logger import get_logger


# ─────────────────────────────────────────────────────────────
# Config / assets
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

LOGO_DARK = ASSETS_DIR / "logo1.png"
TAB_ICON = str(LOGO_DARK) if LOGO_DARK.exists() else "🛡️"

st.set_page_config(
    page_title="ChatGuard",
    page_icon=TAB_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = get_logger(__name__)
config = load_config()


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
}

.block-container {
    padding-top: 1.8rem;
    max-width: 1180px;
}

.chat-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: -4px;
}

.badge-high, .badge-medium, .badge-low {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-top: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
}

.badge-low {
    background: #16351f;
    color: #4ade80;
    border: 1px solid #4ade80;
}

.badge-medium {
    background: #3d2b09;
    color: #fb923c;
    border: 1px solid #fb923c;
}

.badge-high {
    background: #3b1111;
    color: #f87171;
    border: 1px solid #f87171;
}

.mini-metric {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}

.mini-metric .label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.mini-metric .value {
    font-size: 1.55rem;
    font-weight: 700;
}

.analysis-box {
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    background: #0b1220;
    margin-top: 0.5rem;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}

.alert-line {
    border-left: 4px solid #f97316;
    background: #1c1917;
    padding: 0.7rem 0.9rem;
    border-radius: 0 8px 8px 0;
    margin: 0.45rem 0;
}

.stChatMessage {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)


def split_sources_section(text: str) -> tuple[str, list[str]]:
    match = re.search(r"(?im)^sources\s*:\s*$", text)
    if not match:
        return text.strip(), []

    body = text[:match.start()].strip()
    sources_block = text[match.end():].strip()

    source_lines = []
    for raw_line in sources_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        if line:
            source_lines.append(line)

    return body, source_lines


def render_sources(sources: list[str]) -> None:
    if not sources:
        return

    st.markdown("**Sources**")
    for item in sources:
        if item.lower() == "aucune source fiable trouvée":
            st.markdown("- Aucune source fiable trouvée")
            continue

        url_match = URL_RE.search(item)
        if url_match:
            url = url_match.group(0).rstrip(".,;)")
            label = item.replace(url_match.group(0), "").strip(" -–—:|") or url
            st.markdown(f"- [{label}]({url})")
        else:
            st.markdown(f"- {item}")


def render_response_content(text: str) -> None:
    if not text:
        st.write("")
        return

    body, sources = split_sources_section(text)

    body = re.sub(r"\\\((.*?)\\\)", r"$\1$", body, flags=re.DOTALL)
    parts = re.split(r"(\\\[[\s\S]*?\\\])", body)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith(r"\[") and part.endswith(r"\]"):
            expr = part[2:-2].strip()
            if expr:
                st.latex(expr)
        else:
            st.markdown(part)

    render_sources(sources)


def build_sourced_prompt(user_prompt: str) -> str:
    return f"""
Tu réponds en français.

Consignes obligatoires :
- Réponds à la question de façon claire et concise.
- À la fin de chaque réponse, ajoute obligatoirement une section exactement sous cette forme :

Sources :
- https://source-1
- https://source-2
- https://source-3

Règles strictes :
- Chaque source doit être un lien direct complet commençant par http:// ou https://.
- N'écris pas seulement le nom du site ; donne l'URL complète consultable par l'utilisateur.
- Cite uniquement des sources réelles, vérifiables et pertinentes.
- N'invente jamais de source, de lien, de site, de document ou de référence.
- Si tu n'as pas trouvé de source fiable, écris exactement :

Sources :
- Aucune source fiable trouvée

Question :
{user_prompt}
""".strip()


def reliability_badge_html(risk_level: str) -> str:
    if risk_level == "faible":
        return '<span class="badge-low">FIABILITÉ ÉLEVÉE</span>'
    if risk_level == "moyen":
        return '<span class="badge-medium">FIABILITÉ MOYENNE</span>'
    return '<span class="badge-high">FIABILITÉ FAIBLE</span>'


def to_serializable_matrix(matrix):
    if matrix is None:
        return None
    try:
        return matrix.tolist()
    except Exception:
        return matrix


def compact_error_message(error_text: str) -> str:
    text = str(error_text).strip()

    if "rate limit" in text.lower() or "429" in text:
        return (
            "Erreur API : quota ou limite de requêtes atteint. "
            "Réduis le nombre de générations, change de modèle, ou attends le reset du quota."
        )

    if len(text) > 500:
        return text[:500] + "..."
    return text


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_title" not in st.session_state:
    st.session_state.conversation_title = "Nouvelle conversation"


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    provider_labels = ["ChatGPT", "Claude", "Groq", "Gemini"]

    provider_choice = st.selectbox(
        "Provider LLM",
        options=provider_labels,
        index=2,
    )

    groq_model_mapping = {
        "ChatGPT": "groq/compound",
        "Claude": "llama-3.3-70b-versatile",
        "Groq": "groq/compound-mini",
        "Gemini": "llama-3.1-8b-instant",
    }

    model_name = groq_model_mapping[provider_choice]

    if provider_choice == "ChatGPT":
        use_web_search = True
        force_sources = True
    elif provider_choice == "Claude":
        use_web_search = True
        force_sources = True
    elif provider_choice == "Groq":
        use_web_search = True
        force_sources = True
    else:
        use_web_search = False
        force_sources = False

    st.markdown("---")

    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_title = "Nouvelle conversation"
        st.rerun()

# Valeurs fixes
n_generations = 3
temperature = 0.7
alpha = float(config.get("alpha", 0.4))
beta = float(config.get("beta", 0.6))

heuristics_enabled = {
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


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

col_logo, col_title = st.columns([1.2, 5])

with col_logo:
    if LOGO_DARK.exists():
        st.image(str(LOGO_DARK), width=120)

with col_title:
    st.markdown(
        "<h1 style='margin: 0; font-size: 3rem;'>ChatGuard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#94a3b8; font-size:1rem; margin-top:8px;'>"
        "Chatbot avec score de fiabilité sur chaque réponse"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Affichage historique
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_response_content(msg["content"])
            st.markdown(reliability_badge_html(msg["risk_level"]), unsafe_allow_html=True)
            st.caption(
                f"Fiabilité calculée le {msg['timestamp']} · "
                f"Score final {msg['final_score']:.0f}/100"
            )

            with st.expander("Voir l’analyse détaillée"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
<div class="mini-metric">
    <div class="label">Cohérence</div>
    <div class="value">{msg['coherence_score']:.0f}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
<div class="mini-metric">
    <div class="label">Risque heuristique</div>
    <div class="value">{msg['heuristic_score']:.0f}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
<div class="mini-metric">
    <div class="label">Score final</div>
    <div class="value">{msg['final_score']:.0f}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
<div class="analysis-box">
    <div class="section-title">Interprétation</div>
    <div>{msg['justification']}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                alerts = msg.get("alerts", [])
                if alerts:
                    st.markdown("**Alertes détectées**")
                    for alert in alerts:
                        st.markdown(
                            f"""
<div class="alert-line">
    <strong>{alert['name']}</strong> — {alert['explanation']}
</div>
""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("Aucune alerte heuristique déclenchée.")

                contradictions = msg.get("contradictions", [])
                if contradictions:
                    st.markdown("**Contradictions détectées**")
                    for c in contradictions:
                        st.markdown(f"- {c}")

                alternatives = msg.get("alternatives", [])
                if alternatives:
                    with st.expander(f"Réponses alternatives ({len(alternatives)})"):
                        for j, alt in enumerate(alternatives, 1):
                            st.markdown(f"**Génération {j + 1}**")
                            render_response_content(alt)

                matrix = msg.get("similarity_matrix")
                if matrix is not None and len(matrix) >= 2:
                    st.markdown("**Matrice de similarité**")
                    df_sim = pd.DataFrame(
                        matrix,
                        columns=[f"R{i+1}" for i in range(len(matrix))],
                        index=[f"R{i+1}" for i in range(len(matrix))],
                    )
                    try:
                        import matplotlib  # noqa: F401
                        st.dataframe(
                            df_sim.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                            use_container_width=True,
                        )
                    except ImportError:
                        st.dataframe(df_sim.style.format("{:.2f}"), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────
prompt = st.chat_input("Écris ton message...")

if prompt:
    if st.session_state.conversation_title == "Nouvelle conversation":
        st.session_state.conversation_title = prompt[:45] + ("..." if len(prompt) > 45 else "")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        with st.spinner("ChatGuard analyse la réponse..."):
            provider = get_llm_provider(
                provider_name="groq",
                model=model_name,
                temperature=temperature,
                max_tokens=None,
                use_web_search=use_web_search,
            )

            generator = MultiResponseGenerator(provider)

            llm_prompt = build_sourced_prompt(prompt) if force_sources else prompt
            responses = generator.generate(llm_prompt, n=n_generations)

            if not responses:
                raise RuntimeError("Aucune réponse n'a été générée par le modèle.")

            main_response = responses[0]
            alt_responses = responses[1:]

            coherence_analyzer = CoherenceAnalyzer()
            coherence_result = coherence_analyzer.analyze(responses)

            heuristic_detector = HeuristicDetector(enabled=heuristics_enabled)
            heuristic_result = heuristic_detector.analyze(main_response, prompt)

            aggregator = ScoreAggregator(alpha=alpha, beta=beta)
            final_result = aggregator.compute(coherence_result, heuristic_result)

            assistant_message = {
                "role": "assistant",
                "content": main_response,
                "risk_level": final_result.risk_level,
                "coherence_score": float(coherence_result.score),
                "heuristic_score": float(heuristic_result.risk_score),
                "final_score": float(final_result.final_score),
                "justification": final_result.justification,
                "alerts": [
                    {
                        "name": alert.name,
                        "severity": alert.severity,
                        "explanation": alert.explanation,
                    }
                    for alert in heuristic_result.alerts
                ],
                "contradictions": list(coherence_result.contradictions) if coherence_result.contradictions else [],
                "alternatives": alt_responses,
                "similarity_matrix": to_serializable_matrix(coherence_result.similarity_matrix),
                "timestamp": datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
            }

            st.session_state.messages.append(assistant_message)

            logger.info(
                "Chat message analyzed",
                extra={
                    "prompt": prompt[:80],
                    "coherence": coherence_result.score,
                    "heuristic_risk": heuristic_result.risk_score,
                    "final_risk": final_result.final_score,
                    "risk_level": final_result.risk_level,
                },
            )

        st.rerun()

    except Exception as e:
        logger.exception("Chat analysis error")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Erreur pendant l'analyse : {compact_error_message(e)}",
                "risk_level": "élevé",
                "coherence_score": 0.0,
                "heuristic_score": 100.0,
                "final_score": 100.0,
                "justification": "L'analyse n'a pas pu être menée correctement.",
                "alerts": [
                    {
                        "name": "Erreur système",
                        "severity": "high",
                        "explanation": compact_error_message(e),
                    }
                ],
                "contradictions": [],
                "alternatives": [],
                "similarity_matrix": None,
                "timestamp": datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
            }
        )
        st.rerun()