import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "personalize.py"
spec = importlib.util.spec_from_file_location("personalize", SCRIPT)
personalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(personalize)

# Built from the module's constants so this file never contains the literal placeholders itself
# (otherwise `personalize.py` would rewrite - or `--check` would flag - the test file).
USER, NAME = personalize.USER_PLACEHOLDER, personalize.NAME_PLACEHOLDER


def test_placeholders_are_replaced_in_text_files_only(tmp_path):
    (tmp_path / "README.md").write_text(f"see github.com/{USER}/autoinsight", encoding="utf-8")
    (tmp_path / "LICENSE").write_text(f"Copyright {NAME}", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG " + USER.encode())
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "keep.txt").write_text(USER, encoding="utf-8")

    changed = personalize.replace_placeholders(tmp_path, "octocat", "Ada Lovelace")

    assert sorted(changed) == ["LICENSE", "README.md"]
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "see github.com/octocat/autoinsight"
    assert (tmp_path / "LICENSE").read_text(encoding="utf-8") == "Copyright Ada Lovelace"
    assert (tmp_path / "runs" / "keep.txt").read_text(encoding="utf-8") == USER  # skipped dir
    assert personalize.remaining(tmp_path) == []
    assert personalize.replace_placeholders(tmp_path, "octocat", "Ada Lovelace") == []  # idempotent


def test_github_username_validation():
    ok = ["octocat", "a", "my-name-1"]
    bad = ["-bad", "bad-", "has space", "a/b", "x" * 40]
    assert all(personalize.GITHUB_USER.match(u) for u in ok)
    assert not any(personalize.GITHUB_USER.match(u) for u in bad)
