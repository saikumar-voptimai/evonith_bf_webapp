"""Guards that active docs point at canonical post-cleanup paths."""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DOC_ROOTS = [
    REPO_ROOT / "docs",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / ".devcontainer" / "devcontainer.json",
]
COMPATIBILITY_CONTEXT = {
    "archive",
    "compatibility",
    "compatible",
    "deprecated",
    "fallback",
    "legacy",
    "migration",
    "rollback",
    "shim",
    "temporary",
}
SRC_STORAGE_CONTEXT = {
    "archive",
    "cleanup",
    "compatibility",
    "fallback",
    "legacy",
    "migration",
    "moved",
    "old",
    "removed",
    "source",
}
LEGACY_MIGRATION_CONTEXT = {
    "absence",
    "absent",
    "cleanup",
    "dependency",
    "do not reintroduce",
    "must not include",
    "removed",
}


def _active_doc_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_DOC_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".md", ".json"}
                and "archive" not in path.relative_to(REPO_ROOT).parts
            )
    return sorted(files)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _line_window(lines: list[str], index: int, radius: int = 3) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end]).lower()


def _find_lines(pattern: str) -> list[tuple[Path, int, str, str]]:
    regex = re.compile(pattern, re.IGNORECASE)
    matches: list[tuple[Path, int, str, str]] = []
    for path in _active_doc_files():
        lines = _read(path).splitlines()
        for index, line in enumerate(lines):
            if regex.search(line):
                matches.append((path, index + 1, line, _line_window(lines, index)))
    return matches


def _format(matches: list[tuple[Path, int, str, str]]) -> list[str]:
    return [f"{path.relative_to(REPO_ROOT)}:{line_no}: {line}" for path, line_no, line, _ in matches]


def test_readme_mentions_canonical_app_and_package_paths() -> None:
    readme = _read(REPO_ROOT / "README.md")

    assert "apps/backend_api" in readme
    assert "apps/frontend_streamlit" in readme
    assert "packages/furnace-data/furnace_data" in readme
    assert "runtime" in readme


def test_active_docs_do_not_reference_removed_neon_db_package() -> None:
    assert _format(_find_lines(r"furnace_data\.neon_db")) == []


def test_active_docs_do_not_describe_legacy_migration_system_as_active() -> None:
    legacy_name = "ale" + "mbic"
    findings = [
        match
        for match in _find_lines(legacy_name)
        if not any(token in match[3] for token in LEGACY_MIGRATION_CONTEXT)
    ]

    assert _format(findings) == []


def test_active_docs_do_not_describe_src_storage_as_runtime_path() -> None:
    findings = [
        match
        for match in _find_lines(r"src/storage")
        if match[0].relative_to(REPO_ROOT).as_posix()
        != "docs/migration/post-phase-13-structure-cleanup-plan.md"
        and not any(token in match[3] for token in SRC_STORAGE_CONTEXT)
    ]

    assert _format(findings) == []


def test_old_frontend_command_only_appears_in_compatibility_context() -> None:
    findings = [
        match
        for match in _find_lines(r"streamlit run src/app\.py")
        if not any(token in match[3] for token in COMPATIBILITY_CONTEXT)
    ]

    assert _format(findings) == []


def test_primary_docs_do_not_use_legacy_frontend_command() -> None:
    for relative in ("README.md", "docs/deployment/local-install-guide.md", "docs/deployment/local-staging-deployment-guide.md"):
        text = _read(REPO_ROOT / relative)
        canonical = text.find("apps/frontend_streamlit/app.py")
        legacy = text.find("streamlit run src/app.py")
        assert canonical != -1, relative
        if legacy != -1:
            assert canonical < legacy, relative


def test_active_docs_do_not_contain_obvious_real_secret_values() -> None:
    secret_patterns = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",
        r"EVONITH_AUTH_SECRET_KEY=(?!dev-only-secret-change-me|<set-a-strong-random-secret>)[^\s`]+",
    ]
    findings: list[tuple[Path, int, str, str]] = []
    for pattern in secret_patterns:
        findings.extend(_find_lines(pattern))

    assert _format(findings) == []


def test_migration_history_docs_are_archived_outside_active_tree() -> None:
    archive = REPO_ROOT / "docs" / "archive" / "migration-history"

    assert (archive / "migration" / "phase-1-runtime-data.md").is_file()
    assert (archive / "testing" / "phase-12-testing-guide.md").is_file()
    assert not any((REPO_ROOT / "docs" / "migration").glob("phase-*.md"))
    assert not any(
        path.name != "phase-13-testing-guide.md"
        for path in (REPO_ROOT / "docs" / "testing").glob("phase-*.md")
    )
