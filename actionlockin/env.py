"""Project .env loading (zero-dependency). API credentials live ONLY in the gitignored root
.env under their standard names (OPENAI_API_KEY, OPENAI_BASE_URL, DEEPSEEK_API_KEY, ...);
code must not read per-provider keyfiles or relay/alias variables. Variables already
exported in the shell take precedence over .env values."""
from __future__ import annotations
import os
import pathlib

_ROOT_ENV = pathlib.Path(__file__).resolve().parent.parent / ".env"


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ; existing variables win."""
    p = pathlib.Path(path) if path is not None else _ROOT_ENV
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v


def require_env(name: str, path: str | os.PathLike | None = None) -> str:
    """Return a required variable, loading .env first; fail loudly with the fix if unset."""
    load_env(path)
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(
            f"{name} is not set. Put '{name}=...' in the project .env "
            f"(gitignored; see .env.example) or export it in the shell.")
    return v
