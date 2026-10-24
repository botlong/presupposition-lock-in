"""actionlockin.env: API credentials live ONLY in the gitignored project .env under standard
names (OPENAI_API_KEY, ...) — no per-provider keyfiles, no relay/alias variables in code.
Loader is zero-dependency so the package keeps its minimal dependency set."""
import os
import pytest
from actionlockin.env import load_env, require_env


def _absent(monkeypatch, *names):
    # Register each key with monkeypatch before deletion so teardown restores process state.
    for n in names:
        monkeypatch.setenv(n, "_shadow_")
        monkeypatch.delenv(n)


def test_load_env_parses_values_strips_quotes_and_comments(tmp_path, monkeypatch):
    _absent(monkeypatch, "T_A_KEY", "T_B_KEY", "T_C_KEY")
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "\n"
        "T_A_KEY=plain\n"
        "T_B_KEY='single quoted'\n"
        'T_C_KEY="double quoted"\n'
        "not a key-value line\n"
    )
    load_env(env)
    assert os.environ["T_A_KEY"] == "plain"
    assert os.environ["T_B_KEY"] == "single quoted"
    assert os.environ["T_C_KEY"] == "double quoted"


def test_load_env_does_not_override_shell_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("T_A_KEY", "from_shell")
    env = tmp_path / ".env"
    env.write_text("T_A_KEY=from_file\n")
    load_env(env)
    assert os.environ["T_A_KEY"] == "from_shell"


def test_load_env_missing_file_is_a_noop(tmp_path):
    load_env(tmp_path / "absent.env")  # must not raise


def test_require_env_returns_value_from_env_file(tmp_path, monkeypatch):
    _absent(monkeypatch, "T_SOME_API_KEY")
    env = tmp_path / ".env"
    env.write_text("T_SOME_API_KEY=abc123\n")
    assert require_env("T_SOME_API_KEY", path=env) == "abc123"


def test_require_env_fails_loudly_when_unset(tmp_path, monkeypatch):
    _absent(monkeypatch, "T_SOME_API_KEY")
    with pytest.raises(RuntimeError, match="T_SOME_API_KEY"):
        require_env("T_SOME_API_KEY", path=tmp_path / "absent.env")
