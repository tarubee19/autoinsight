"""Fill in the two placeholders before you publish the repository (works on Windows, macOS and Linux).

    python scripts/personalize.py --github-user your-github-name --name "Your Full Name"
    python scripts/personalize.py --check          # list placeholders that are still there

Replaces ``tarubee19`` (README, badges, deploy files, docs) and ``taroon`` (LICENSE).
Safe to run more than once; only text files are touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "runs", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".joblib", ".zip", ".pyc"}
USER_PLACEHOLDER = "tarubee19"
NAME_PLACEHOLDER = "taroon"
GITHUB_USER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


def text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(root).parts) or path.resolve() == Path(__file__).resolve():
            continue
        yield path


def replace_placeholders(root: Path, github_user: str | None, name: str | None) -> list[str]:
    """Replace the placeholders under ``root``; returns the relative paths that changed."""
    changed = []
    for path in text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        if github_user:
            new = new.replace(USER_PLACEHOLDER, github_user)
        if name:
            new = new.replace(NAME_PLACEHOLDER, name)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(path.relative_to(root).as_posix())
    return changed


def remaining(root: Path) -> list[str]:
    found = []
    for path in text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in (USER_PLACEHOLDER, NAME_PLACEHOLDER):
            if marker in text:
                found.append(f"{path.relative_to(root).as_posix()}: {marker}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--github-user", help="your GitHub username (replaces tarubee19)")
    parser.add_argument("--name", help="your full name for the licence (replaces taroon)")
    parser.add_argument("--check", action="store_true", help="only list placeholders that are still present")
    args = parser.parse_args(argv)

    if args.check:
        left = remaining(ROOT)
        print("\n".join(left) if left else "No placeholders left.")
        return 1 if left else 0
    if not args.github_user and not args.name:
        parser.error("give --github-user and/or --name (or use --check)")
    if args.github_user and not GITHUB_USER.match(args.github_user):
        parser.error("--github-user must be a valid GitHub username (letters, digits, single hyphens)")

    for rel in replace_placeholders(ROOT, args.github_user, args.name):
        print("updated", rel)
    left = remaining(ROOT)
    print("\nStill to fill in:\n" + "\n".join(left) if left else "\nAll placeholders are filled in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
