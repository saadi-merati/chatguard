# 🛡️ ChatGuard

**Détecteur intelligent de réponses à risque générées par un chatbot LLM**

ChatGuard est un prototype de recherche qui analyse la fiabilité et le risque des réponses générées par un LLM. Il combine deux approches complémentaires : la **cohérence interne** (analyse sémantique de plusieurs générations) et des **heuristiques explicables** (règles transparentes de détection de risque).

---

## 📸 Aperçu

```
Prompt utilisateur  →  Génération multiple (N réponses)
                              ↓
              ┌───────────────┴───────────────┐
              │                               │
     Cohérence interne              Heuristiques de risque
     (embeddings cosinus)           (règles lexicales)
              │                               │
              └───────────────┬───────────────┘
                              ↓
              Score final = α × incohérence + β × risque_heuristique
                              ↓
                     Risque FAIBLE / MOYEN / ÉLEVÉ
```

---

## 🗂️ Structure du projet

```
chatguard/
├── app.py                        # Interface Streamlit principale
├── requirements.txt
├── .env.example
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Chargement .env, valeurs par défaut
│
├── llm/
│   ├── __init__.py
│   ├── provider.py               # Couche d'abstraction LLM (OpenAI, HF, Mock)
│   └── generator.py              # Génération multiple de réponses
│
├── analysis/
│   ├── __init__.py
│   └── coherence.py              # Analyse de cohérence sémantique
│
├── heuristics/
│   ├── __init__.py
│   └── detector.py               # 9 heuristiques de détection de risque
│
├── scoring/
│   ├── __init__.py
│   └── aggregator.py             # Fusion pondérée des scores
│
├── data/
│   └── test_prompts.jsonl        # Corpus de 25 prompts de test
│
├── utils/
│   ├── __init__.py
│   └── logger.py                 # Logging centralisé
│
├── tests/
│   ├── __init__.py
│   └── test_all.py               # Tests unitaires (pytest)
│
└── logs/
    └── chatguard.log             # Généré automatiquement
```

---

## ⚙️ Installation

### Prérequis
- Python 3.11+
- pip
- (Optionnel) clé API OpenAI ou token HuggingFace

### Étapes

```bash
# 1. Cloner ou décompresser le projet
cd chatguard/

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# ou : venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env
# Éditez .env avec votre clé API OpenAI ou token HuggingFace

# 5. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

### Mode démo sans API (recommandé pour commencer)

Sélectionnez **"mock"** dans la sidebar comme provider LLM.  
Aucune clé API n'est requise. Des réponses pré-définies simulent différents scénarios.

---

## 🔑 Configuration des providers

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo    # ou gpt-4, gpt-4o
```

### HuggingFace Inference API

```env
LLM_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=hf_...
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
```

Le token HuggingFace est gratuit pour les modèles publics (avec limite de taux).  
Obtenez-le sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Mock (tests / démo)

```env
LLM_PROVIDER=mock
```

---

## 📐 Logique de scoring

### Formule

```
incoherence_risk  = 100 - coherence_score
final_risk        = α × incoherence_risk + β × heuristic_risk_score
```

### Paramètres par défaut

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| α (alpha) | 0.4 | Poids de l'incohérence interne |
| β (beta)  | 0.6 | Poids du risque heuristique |

**Justification du choix α < β :**  
L'analyse heuristique est plus déterministe et directement interprétable. La cohérence interne, bien qu'informative, dépend de la température et du nombre de générations. On lui donne donc un poids légèrement inférieur.

### Seuils d'interprétation

| Score | Niveau de risque | Signification |
|-------|-----------------|---------------|
| 0–33  | 🟢 Faible        | Réponse probablement fiable |
| 34–66 | 🟡 Moyen         | Vérification conseillée |
| 67–100 | 🔴 Élevé        | Vérification obligatoire |

### Exemple commenté

```
Prompt : "Quelle est la dose maximale de paracétamol par jour ?"

Génération : 3 réponses (température = 0.7)
  R1 : "La dose maximale est de 4g/jour, soit 8 comprimés de 500mg..."
  R2 : "En général entre 3g et 4g selon les sources médicales..."
  R3 : "Cela dépend du patient, environ 3 à 4 grammes..."

Cohérence interne :
  Similarité R1-R2 : 0.82
  Similarité R1-R3 : 0.78
  Similarité R2-R3 : 0.84
  Moyenne : 0.81 → cohérence = 81/100

Heuristiques déclenchées :
  [HIGH]   Sujet sensible : santé / médical → +25 pts
  [MEDIUM] Vocabulaire d'incertitude (selon les sources, environ) → +15 pts
  [MEDIUM] Absence de source sur données chiffrées → +15 pts
  Score heuristique brut : 55/100

Calcul final :
  incoherence_risk = 100 - 81 = 19
  final_risk = 0.4 × 19 + 0.6 × 55 = 7.6 + 33 = 40.6

→ Risque MOYEN (40.6/100)
→ "Quelques signaux d'alerte détectés. Croiser avec d'autres sources."
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Avec couverture de code
python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Un seul test
python -m pytest tests/test_all.py::TestHeuristicDetector::test_medical_topic_detected -v
```

### Couverture des tests

| Module | Tests |
|--------|-------|
| `llm/provider.py` | Provider Mock, Factory, repr |
| `llm/generator.py` | Génération N réponses, métadonnées |
| `analysis/coherence.py` | Réponses identiques, contradictoires, matrice |
| `heuristics/detector.py` | 6 heuristiques individuelles, désactivation |
| `scoring/aggregator.py` | Formule numérique, classification, clamp |
| `config/settings.py` | Chargement, clés requises |
| `data/` | Corpus JSONL, champs requis |

---

## 📊 Données et corpus

### Type de données

Le fichier `data/test_prompts.jsonl` contient 25 prompts de test répartis en 6 catégories :

| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| `medical` | 5 | Questions médicales sensibles |
| `legal` | 2 | Questions juridiques |
| `finance` | 2 | Conseils financiers |
| `factual` | 6 | Faits vérifiables simples |
| `ambiguous` | 6 | Questions ouvertes ou controversées |
| `hallucination_trap` | 4 | Pièges à hallucination (études fictives, ISBN…) |
| `sensitive` | 3 | Contenus potentiellement dangereux |

### Source des données

Les prompts sont **synthétiques**, conçus manuellement pour couvrir les cas d'usage typiques de ChatGuard. Ils s'inspirent de :
- Benchmarks publics : [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [HaluEval](https://github.com/RUCAIBox/HaluEval)
- Exemples de prompt injection et red-teaming documentés

### Mode d'accès

```python
import json
with open("data/test_prompts.jsonl") as f:
    prompts = [json.loads(line) for line in f if line.strip()]
```

### Enrichissement futur

Pour ajouter des prompts au corpus :
1. Ouvrir `data/test_prompts.jsonl`
2. Ajouter une ligne JSON avec les champs : `id`, `category`, `risk`, `prompt`, `expected_risk`, `notes`
3. Incrémenter l'`id`

Pour un corpus plus large, vous pouvez utiliser :
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) (817 questions + réponses de référence)
- [BOLD](https://huggingface.co/datasets/AlexaAI/bold) (biais et stéréotypes)
- [WinoBias](https://huggingface.co/datasets/wino_bias) (biais de genre)

---

## 🔬 Heuristiques détaillées

| Heuristique | Sévérité | Poids | Déclencheur |
|-------------|----------|-------|-------------|
| Vocabulaire d'incertitude | medium | +15 | ≥3 marqueurs : "peut-être", "il semble"… |
| Langage vague | medium | +15 | ≥2 patterns vagues |
| Absence de source | medium | +15 | Données chiffrées sans référence |
| Affirmations non justifiées | high | +25 | ≥3 chiffres sans source |
| Réponse trop courte | medium | +15 | < 20 mots |
| Sujet sensible | high | +25 | Santé, droit, finance, sécurité |
| Contradiction interne | high | +25 | Paires "toujours/jamais", "légal/illégal"… |
| Formulations absolues | medium | +15 | ≥2 absolus : "toujours", "certainement"… |
| Manque de nuance | low | +8 | Pas de connecteur logique, ≤3 phrases |

---

## 🧠 Analyse de cohérence

### Méthode principale : sentence-transformers + cosinus

1. Encoder chaque réponse Rᵢ avec `all-MiniLM-L6-v2` → vecteur de 384 dimensions
2. Calculer la matrice de similarité cosinus NxN
3. Extraire les similarités off-diagonal (paires uniques)
4. Score de cohérence = `0.6 × mean_similarity + 0.4 × min_similarity` × 100
5. Pénalité lexicale : -5 pts par paire de contraires détectée

### Méthode de secours (sans sentence-transformers)

Recouvrement lexical (Jaccard) entre les ensembles de mots de chaque paire de réponses.

### Détection de contradictions

Une paire est marquée comme contradictoire si sa similarité cosinus < 0.65.

---

## 🚀 Améliorations futures

### Court terme
- [ ] Intégration d'un modèle NLI (ex: `cross-encoder/nli-deberta-v3-small`) pour détection de contradictions plus précise
- [ ] Export CSV des analyses pour audit
- [ ] Historique des analyses en session
- [ ] Mode batch : analyser tout le corpus `test_prompts.jsonl` automatiquement

### Moyen terme
- [ ] Intégration de la **perplexité** comme signal de confiance supplémentaire
- [ ] Fact-checking automatique via API (Wikipedia, Wikidata)
- [ ] Détection de toxicité (Perspective API, Detoxify)
- [ ] Support multilingue pour les heuristiques (EN, DE, ES…)

### Long terme
- [ ] Fine-tuning d'un classificateur de risque sur des exemples annotés
- [ ] Détection de prompt injection et d'adversarial prompts
- [ ] Interface API REST (FastAPI) pour intégration dans d'autres systèmes
- [ ] Dashboard de monitoring avec métriques agrégées

---

## ⚠️ Limites du prototype

1. **Dépendance à la température** : la cohérence interne varie avec la température. À T=0, toutes les réponses seront identiques → score de cohérence = 100 artificiellement.

2. **Heuristiques heuristiques** : les règles sont nécessairement imparfaites. Une réponse médicale bien sourcée déclenchera quand même l'alerte "sujet sensible".

3. **Pas de fact-checking réel** : ChatGuard ne vérifie pas si les informations sont vraies, seulement si elles présentent des signaux de risque formel.

4. **Langue française** : les vocabulaires et patterns sont optimisés pour le français. Une adaptation est nécessaire pour d'autres langues.

5. **Coût API** : générer 3-5 réponses par prompt multiplie le coût par 3-5 (important en production).

6. **Modèle d'embedding** : `all-MiniLM-L6-v2` est entraîné majoritairement en anglais. Pour le français, préférer `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 👥 Crédits

Projet développé dans le cadre d'un prototype académique de détection de risque dans les LLM.

**Bibliothèques utilisées :**
- [Streamlit](https://streamlit.io/) — Interface web
- [sentence-transformers](https://www.sbert.net/) — Embeddings sémantiques
- [OpenAI Python SDK](https://github.com/openai/openai-python) — Provider OpenAI
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Gestion .env
- [pandas](https://pandas.pydata.org/) — Manipulation de données

---

## 📄 Licence

MIT License — Libre d'utilisation à des fins éducatives et de recherche.
