"""Behaviour of recogniser guards that need no model loaded."""

import pytest

from meet.asr import LANGUAGES, echoes_context, resolve_language
from meet.config import FALLBACK_LANGUAGE

GLOSSARY = "Vocabulary: Kubernetes, Postgres, Terraform, gRPC, OAuth, Redis."


class TestEchoesContext:
    def test_detects_the_glossary_returned_as_a_transcript(self):
        """Observed for real: an unclear segment came back as the whole prompt."""
        assert echoes_context(GLOSSARY, GLOSSARY)

    def test_detects_a_partial_glossary_echo(self):
        assert echoes_context("Vocabulary: Kubernetes, Postgres", GLOSSARY)

    def test_real_speech_is_kept(self):
        assert not echoes_context("不 sure if you can hear me because", GLOSSARY)

    def test_speech_mentioning_glossary_terms_is_kept(self):
        """A sentence using the vocabulary must not look like an echo of it."""
        assert not echoes_context("Kubernetes 那边 ingress 的设定有问题", GLOSSARY)

    def test_empty_values_are_not_echoes(self):
        assert not echoes_context("", GLOSSARY)
        assert not echoes_context("anything", "")


class TestLanguageChoices:
    def test_english_is_the_fallback_when_nothing_is_configured(self):
        """Asserts the constant, not the resolved value, which MEET_LANGUAGE moves."""
        assert FALLBACK_LANGUAGE == "en"

    def test_mandarin_and_auto_are_selectable(self):
        assert "zh" in LANGUAGES
        assert "auto" in LANGUAGES

    def test_the_fallback_is_itself_a_valid_choice(self):
        assert FALLBACK_LANGUAGE in LANGUAGES


class TestResolveLanguage:
    def test_accepts_every_supported_language(self):
        for language in LANGUAGES:
            assert resolve_language(language) == language

    def test_rejects_an_unsupported_language(self):
        """MEET_LANGUAGE bypasses argparse choices, so this must be caught."""
        with pytest.raises(ValueError, match="unknown language"):
            resolve_language("de")

    def test_rejects_an_empty_language(self):
        with pytest.raises(ValueError, match="unknown language"):
            resolve_language("")
