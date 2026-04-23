"""Resolve project paths for local runs and Streamlit Cloud (cwd may be repo root)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Directory that contains `app.py`, `data/`, and `outputs/` (the `ai_index_number` folder)."""
    here = Path(__file__).resolve()
    # src/utils/paths.py -> ai_index_number
    return here.parent.parent.parent


def resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidate = project_root() / p
    if candidate.exists():
        return candidate
    return p
