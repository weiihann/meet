"""Behaviour of note rendering and prompt assembly."""

from datetime import date

from meet.glossary import context_prompt, parse_terms
from meet.summarize import build_prompt, note_path, render_note, safe_title

WHEN = date(2026, 8, 17)


class TestSafeTitle:
    def test_strips_characters_that_break_filenames(self):
        assert safe_title("BSC/opBNB: bridge?") == "BSC-opBNB- bridge-"

    def test_never_returns_an_empty_filename(self):
        assert safe_title("///") == "-"
        assert safe_title("   ") == "Untitled"

    def test_keeps_mandarin_intact(self):
        assert safe_title("跨链 bridge 讨论") == "跨链 bridge 讨论"


class TestNotePath:
    def test_filename_uses_the_vault_s_compact_date(self):
        """The vault's own notes are named "20260817 ...", with no dashes."""
        assert note_path(WHEN, "Bridge sync").endswith("/20260817 Meeting - Bridge sync.md")

    def test_lands_in_the_notes_folder(self):
        assert "/Notes/" in note_path(WHEN, "x")

    def test_sanitises_the_title_into_the_filename(self):
        assert note_path(WHEN, "BSC/opBNB").endswith("20260817 Meeting - BSC-opBNB.md")


class TestRenderNote:
    def test_matches_the_vault_meeting_template_frontmatter(self):
        note = render_note(WHEN, "Bridge sync", "## Notes\n- a", "[00:00] **Me:** hi")
        assert note.startswith(
            '---\ncreated: "2026-08-17"\norg: \ntags:\n  - note\n  - meeting\n---'
        )

    def test_embeds_the_transcript_as_a_collapsed_callout(self):
        note = render_note(WHEN, "T", "body", "line one\nline two")
        assert "> [!note]- Full transcript" in note
        assert "> line one" in note
        assert "> line two" in note

    def test_carries_the_org_field_through(self):
        assert "org: BNB Chain" in render_note(WHEN, "T", "b", "t", org="BNB Chain")


class TestBuildPrompt:
    def test_includes_transcript_and_glossary(self):
        prompt = build_prompt("[00:00] **Me:** hello", "Parlia, opBNB")
        assert "hello" in prompt
        assert "Parlia, opBNB" in prompt

    def test_states_none_when_glossary_is_empty(self):
        assert "(none)" in build_prompt("x", "")

    def test_instructs_the_model_not_to_translate_mandarin(self):
        assert "Keep Mandarin in Mandarin" in build_prompt("x", "y")


class TestGlossaryParsing:
    def test_ignores_comments_and_blank_lines(self):
        assert parse_terms("# header\n\nParlia\n  BEP  \n# trailing") == ["Parlia", "BEP"]

    def test_context_prompt_is_empty_when_there_are_no_terms(self):
        assert context_prompt([]) == ""

    def test_context_prompt_lists_terms_for_the_recogniser(self):
        assert context_prompt(["Parlia", "opBNB"]) == "Vocabulary: Parlia, opBNB."
