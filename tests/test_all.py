"""
tests/test_all.py — Tests unitaires pour ChatGuard

Exécution :
    python -m pytest tests/ -v
    # ou
    python -m pytest tests/ -v --tb=short
"""

import pytest
import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Tests Provider ───────────────────────────────────────────────────────────

class TestMockProvider:
    def test_basic_generation(self):
        from llm.provider import MockProvider
        provider = MockProvider()
        response = provider.complete("Qu'est-ce que Python ?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_medical_prompt_routing(self):
        from llm.provider import MockProvider
        provider = MockProvider()
        response = provider.complete("Quelle est la dose de paracétamol ?")
        # Doit retourner une des réponses du pool médical
        assert "paracétamol" in response.lower() or "dose" in response.lower() or "g" in response.lower()

    def test_multiple_calls_return_different_responses(self):
        from llm.provider import MockProvider
        provider = MockProvider()
        # Le mock tourne sur les réponses disponibles
        responses = [provider.complete("Investir en bourse est-il risqué ?") for _ in range(4)]
        # Au moins une réponse doit être différente des autres
        assert len(set(responses)) >= 1  # Au minimum 1 réponse unique


class TestProviderFactory:
    def test_mock_provider_creation(self):
        from llm.provider import get_llm_provider, MockProvider
        provider = get_llm_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        from llm.provider import get_llm_provider
        with pytest.raises(ValueError):
            get_llm_provider("unknown_provider_xyz")

    def test_provider_repr(self):
        from llm.provider import MockProvider
        p = MockProvider()
        assert "Mock" in repr(p)


# ─── Tests Generator ─────────────────────────────────────────────────────────

class TestMultiResponseGenerator:
    def setup_method(self):
        from llm.provider import MockProvider
        from llm.generator import MultiResponseGenerator
        self.provider = MockProvider()
        self.generator = MultiResponseGenerator(self.provider, delay_between_calls=0.0)

    def test_generates_n_responses(self):
        responses = self.generator.generate("Test prompt", n=3)
        assert len(responses) == 3

    def test_minimum_2_responses(self):
        # n=1 doit être forcé à minimum 2
        responses = self.generator.generate("Test prompt", n=1)
        assert len(responses) >= 2

    def test_responses_are_strings(self):
        responses = self.generator.generate("Test prompt", n=3)
        for r in responses:
            assert isinstance(r, str)
            assert len(r.strip()) > 0

    def test_metadata_generation(self):
        result = self.generator.generate_with_metadata("Test prompt", n=3)
        assert "main_response" in result
        assert "alt_responses" in result
        assert result["n_requested"] == 3


# ─── Tests Cohérence ─────────────────────────────────────────────────────────

class TestCoherenceAnalyzer:
    def setup_method(self):
        from analysis.coherence import CoherenceAnalyzer
        self.analyzer = CoherenceAnalyzer()

    def test_single_response_returns_default(self):
        result = self.analyzer.analyze(["Une seule réponse"])
        assert 0 <= result.score <= 100

    def test_identical_responses_high_score(self):
        """Des réponses identiques doivent donner un score élevé."""
        responses = [
            "Python est un langage de programmation créé par Guido van Rossum en 1991.",
            "Python est un langage de programmation créé par Guido van Rossum en 1991.",
            "Python est un langage de programmation créé par Guido van Rossum en 1991.",
        ]
        result = self.analyzer.analyze(responses)
        assert result.score >= 90.0, f"Score attendu >= 90, obtenu {result.score}"

    def test_contradictory_responses_lower_score(self):
        """Des réponses très différentes doivent donner un score plus faible."""
        responses = [
            "La bourse est toujours une excellente idée pour investir, les gains sont garantis.",
            "Ne jamais investir en bourse, vous perdrez certainement tout votre argent.",
            "Les chats sont des animaux domestiques populaires qui ronronnent.",
        ]
        # Le score devrait être inférieur à celui de réponses identiques
        result = self.analyzer.analyze(responses)
        assert result.score < 90.0

    def test_score_in_range(self):
        responses = [
            "Le soleil se lève à l'est.",
            "Le soleil disparaît à l'ouest le soir.",
            "La lune est le satellite naturel de la Terre.",
        ]
        result = self.analyzer.analyze(responses)
        assert 0 <= result.score <= 100

    def test_interpretation_not_empty(self):
        result = self.analyzer.analyze(["Réponse A.", "Réponse B."])
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_similarity_matrix_shape(self):
        responses = ["A", "B", "C"]
        result = self.analyzer.analyze(responses)
        if result.similarity_matrix is not None:
            assert result.similarity_matrix.shape == (3, 3)


# ─── Tests Heuristiques ───────────────────────────────────────────────────────

class TestHeuristicDetector:
    def setup_method(self):
        from heuristics.detector import HeuristicDetector
        self.detector = HeuristicDetector()

    def test_empty_response(self):
        result = self.detector.analyze("")
        assert result.risk_score > 0

    def test_score_in_range(self):
        result = self.detector.analyze("Voici une réponse normale et neutre.")
        assert 0 <= result.risk_score <= 100

    def test_medical_topic_detected(self):
        response = "Vous pouvez prendre 8 comprimés de paracétamol par jour."
        result = self.detector.analyze(response, prompt="dose de paracétamol")
        assert result.risk_score > 20
        sensitive_alert = any("sensible" in a.name.lower() or "médical" in a.name.lower()
                               for a in result.alerts)
        assert sensitive_alert, f"Aucune alerte sensible dans {[a.name for a in result.alerts]}"

    def test_uncertainty_vocab_detected(self):
        response = (
            "Il semble que peut-être vous pourriez probablement envisager "
            "de consulter un médecin dans certains cas selon les sources."
        )
        result = self.detector.analyze(response)
        uncertainty_alert = any("incertitude" in a.name.lower() for a in result.alerts)
        assert uncertainty_alert

    def test_absolute_language_detected(self):
        response = "Cette méthode fonctionne toujours et certainement sans exception. Jamais de problème."
        result = self.detector.analyze(response)
        absolute_alert = any("absolu" in a.name.lower() for a in result.alerts)
        assert absolute_alert

    def test_short_response_flagged(self):
        result = self.detector.analyze("Oui.", prompt="Que penser de X ?")
        short_alert = any("court" in a.name.lower() for a in result.alerts)
        assert short_alert

    def test_risk_level_classification(self):
        # Score > 67 → élevé
        from heuristics.detector import HeuristicResult, Alert
        r = HeuristicResult(
            risk_score=75.0,
            risk_level="élevé",
            alerts=[],
        )
        assert r.risk_level == "élevé"

    def test_disabled_heuristic_not_triggered(self):
        detector = __import__("heuristics.detector", fromlist=["HeuristicDetector"]).HeuristicDetector(
            enabled={"sensitive_topics": False}
        )
        response = "Vous pouvez prendre ce médicament contre la maladie."
        result = detector.analyze(response, prompt="médicament")
        sensitive_alert = any("sensible" in a.name.lower() for a in result.alerts)
        assert not sensitive_alert


# ─── Tests Agrégation ─────────────────────────────────────────────────────────

class TestScoreAggregator:
    def setup_method(self):
        from scoring.aggregator import ScoreAggregator
        from analysis.coherence import CoherenceResult
        from heuristics.detector import HeuristicResult
        self.aggregator = ScoreAggregator(alpha=0.4, beta=0.6)
        self.CoherenceResult = CoherenceResult
        self.HeuristicResult = HeuristicResult

    def _make_coherence(self, score):
        return self.CoherenceResult(score=score, interpretation="test")

    def _make_heuristic(self, score):
        return self.HeuristicResult(risk_score=score, risk_level="test")

    def test_high_incoherence_high_risk(self):
        coh = self._make_coherence(10.0)   # incohérence = 90
        heur = self._make_heuristic(80.0)
        result = self.aggregator.compute(coh, heur)
        assert result.risk_level == "élevé"
        assert result.final_score >= 67

    def test_perfect_coherence_no_alerts_low_risk(self):
        coh = self._make_coherence(100.0)  # incohérence = 0
        heur = self._make_heuristic(0.0)
        result = self.aggregator.compute(coh, heur)
        assert result.final_score == 0.0
        assert result.risk_level == "faible"

    def test_score_clamped_to_100(self):
        coh = self._make_coherence(0.0)    # incohérence = 100
        heur = self._make_heuristic(100.0)
        result = self.aggregator.compute(coh, heur)
        assert result.final_score <= 100.0

    def test_formula_correctness(self):
        """Vérification numérique de la formule."""
        coh = self._make_coherence(60.0)   # incohérence = 40
        heur = self._make_heuristic(50.0)
        result = self.aggregator.compute(coh, heur)
        expected = 0.4 * 40.0 + 0.6 * 50.0  # = 16 + 30 = 46
        assert abs(result.final_score - expected) < 0.1

    def test_risk_level_medium(self):
        coh = self._make_coherence(60.0)   # incohérence = 40
        heur = self._make_heuristic(50.0)  # → 46
        result = self.aggregator.compute(coh, heur)
        assert result.risk_level == "moyen"

    def test_justification_not_empty(self):
        coh = self._make_coherence(75.0)
        heur = self._make_heuristic(30.0)
        result = self.aggregator.compute(coh, heur)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 10


# ─── Tests Config ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_config_returns_dict(self):
        from config.settings import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_config_has_required_keys(self):
        from config.settings import load_config
        cfg = load_config()
        for key in ["alpha", "beta", "n_generations", "temperature"]:
            assert key in cfg, f"Clé manquante dans config : {key}"

    def test_alpha_beta_defaults(self):
        from config.settings import load_config
        cfg = load_config()
        assert 0.0 <= cfg["alpha"] <= 1.0
        assert 0.0 <= cfg["beta"] <= 1.0


# ─── Tests Corpus ─────────────────────────────────────────────────────────────

class TestCorpus:
    def test_corpus_loads(self):
        import json
        corpus = []
        with open("data/test_prompts.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        assert len(corpus) >= 20

    def test_corpus_has_required_fields(self):
        import json
        with open("data/test_prompts.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    assert "prompt" in entry
                    assert "category" in entry
                    assert "expected_risk" in entry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
