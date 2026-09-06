"""Verify that optional Torch dependencies remain lazily imported.

This was originally a one-shot source rewriter. Keeping a non-idempotent
rewriter in the repository made a second invocation fail (and risked rewriting
the wrong ``utils.py`` when run elsewhere), so it is now a safe maintenance
check.
"""

from __future__ import annotations

from pathlib import Path


UTILS_PATH = Path(__file__).resolve().with_name("utils.py")


def main() -> None:
    """Fail clearly if heavy optional imports drift back to module scope."""
    source = UTILS_PATH.read_text(encoding="utf-8")
    preamble = source.split("def get_missing_nnpot_packages", 1)[0]
    eager_imports = [
        statement
        for statement in ("import torch", "from e3nn", "from nnpot_models")
        if statement in preamble
    ]
    if eager_imports:
        joined = ", ".join(eager_imports)
        raise SystemExit(f"utils.py eagerly imports optional dependencies: {joined}")
    print("utils.py: optional Torch dependencies are imported on demand")


if __name__ == "__main__":
    main()
