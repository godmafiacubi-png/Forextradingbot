"""Lightweight repository secret scanner for CI.

This is intentionally dependency-free. It blocks common private-key blocks and
high-risk credential assignments while allowing placeholders in .env.example.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "venv", ".venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pkl", ".joblib", ".pt", ".pth", ".h5", ".keras", ".png", ".jpg", ".jpeg"}
ALLOW_FILES = {".env.example"}
SKIP_NAMES = {".env"}

PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"]?([^'\"\s#]{16,})"),
    re.compile(r"(?i)(telegram[_-]?token)\s*=\s*['\"]?([^'\"\s#]{10,})"),
]

PLACEHOLDERS = {"changeme", "example", "placeholder", "dummy", "test", "none", "null"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel_parts = set(path.relative_to(ROOT).parts)
        if (
            path.is_dir()
            or rel_parts & SKIP_DIRS
            or path.name in SKIP_NAMES
            or path.name.startswith(".env.")
            or path.suffix.lower() in SKIP_SUFFIXES
        ):
            continue
        files.append(path)
    return files


def is_allowed(path: Path, match: re.Match[str]) -> bool:
    if path.name in ALLOW_FILES or "os.getenv" in match.string:
        return True
    if match.lastindex and match.lastindex >= 2:
        value = match.group(2).strip().strip('"\'').lower()
        return value in PLACEHOLDERS
    return False


def main() -> int:
    findings: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                match = pattern.search(line)
                if match and not is_allowed(path, match):
                    findings.append(f"{path.relative_to(ROOT)}:{line_no}: possible secret")
    if findings:
        print("Potential secrets found:")
        print("\n".join(findings))
        return 1
    print("No obvious secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
