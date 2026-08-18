"""Behaviour of recogniser guards that need no model loaded."""

from meet.asr import DEFAULT_LANGUAGE, LANGUAGES, echoes_context

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
    def test_mandarin_is_the_default_because_it_covers_both(self):
        assert DEFAULT_LANGUAGE == "zh"

    def test_english_and_auto_remain_available(self):
        assert "en" in LANGUAGES
        assert "auto" in LANGUAGES
