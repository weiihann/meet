"""Behaviour of configuration loading."""

from pathlib import Path

from meet.config import apply_env, parse_env, resolve_root


class TestParseEnv:
    def test_reads_simple_assignments(self):
        assert parse_env("MEET_VAULT=/tmp/vault\n") == {"MEET_VAULT": "/tmp/vault"}

    def test_ignores_comments_and_blank_lines(self):
        text = "# a comment\n\nMEET_ROOT=/x\n   # indented comment\n"
        assert parse_env(text) == {"MEET_ROOT": "/x"}

    def test_tolerates_spaces_around_the_equals(self):
        assert parse_env("MEET_ROOT = /x\n") == {"MEET_ROOT": "/x"}

    def test_strips_matched_quotes(self):
        assert parse_env("A=\"/a b\"\nB='/c d'\n") == {"A": "/a b", "B": "/c d"}

    def test_keeps_equals_signs_inside_values(self):
        """Model ids and query strings contain '='; only the first one splits."""
        assert parse_env("MEET_ASR_MODEL=org/model=v2\n") == {"MEET_ASR_MODEL": "org/model=v2"}

    def test_accepts_an_export_prefix(self):
        assert parse_env("export MEET_ROOT=/x\n") == {"MEET_ROOT": "/x"}

    def test_allows_an_empty_value(self):
        assert parse_env("MEET_ORG=\n") == {"MEET_ORG": ""}

    def test_keeps_hash_inside_a_value(self):
        """Not treated as an inline comment, since values may legitimately contain #."""
        assert parse_env("A=va#lue\n") == {"A": "va#lue"}

    def test_skips_lines_with_no_assignment(self):
        assert parse_env("this is not a setting\nA=1\n") == {"A": "1"}

    def test_empty_text_yields_nothing(self):
        assert parse_env("") == {}


class TestApplyEnv:
    def test_sets_variables_that_are_not_already_present(self):
        environ: dict[str, str] = {}
        apply_env({"MEET_ROOT": "/x"}, environ)
        assert environ == {"MEET_ROOT": "/x"}

    def test_never_overrides_the_real_environment(self):
        """`MEET_VAULT=/override meet ...` must beat the .env file."""
        environ = {"MEET_VAULT": "/override"}
        apply_env({"MEET_VAULT": "/from-dotenv"}, environ)
        assert environ["MEET_VAULT"] == "/override"

    def test_reports_how_many_it_applied(self):
        environ = {"A": "kept"}
        assert apply_env({"A": "ignored", "B": "set"}, environ) == 1


class TestResolveRoot:
    def test_uses_the_checkout_when_one_is_detected(self, tmp_path: Path):
        """src/meet/config.py -> the repo root two levels up."""
        repo = tmp_path / "clone"
        (repo / "src" / "meet").mkdir(parents=True)
        (repo / "pyproject.toml").touch()
        module = repo / "src" / "meet" / "config.py"
        assert resolve_root(module, tmp_path / "fallback") == repo

    def test_falls_back_when_not_inside_a_checkout(self, tmp_path: Path):
        """Installed into site-packages, where no pyproject.toml sits above."""
        installed = tmp_path / "venv" / "lib" / "meet"
        installed.mkdir(parents=True)
        fallback = tmp_path / "fallback"
        assert resolve_root(installed / "config.py", fallback) == fallback
