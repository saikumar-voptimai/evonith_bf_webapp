"""Tests for shared page registry metadata."""

from pathlib import Path

from src.config.page_registry import get_feedback_page_options, get_navigation_pages


def test_feedback_options_come_from_navigation_registry() -> None:
    """Feedback dropdown options should mirror navigation titles."""
    descriptors = get_navigation_pages()
    expected_titles = [descriptor.title for descriptor in descriptors if descriptor.include_in_feedback]
    assert get_feedback_page_options() == expected_titles


def test_furnace_status_is_registered_directly_after_welcome() -> None:
    """Furnace Status is one visible nav item, placed right after Welcome."""
    descriptors = get_navigation_pages()
    assert descriptors[0].title == "Welcome"

    status = descriptors[1]
    assert status.file_path == "custom_pages/10_Furnace_Status.py"
    assert status.title == "Furnace Status"
    assert status.icon == "🔥"

    # The trend view is a query-parameter view of the same page, never a second item.
    assert [d.file_path for d in descriptors].count("custom_pages/10_Furnace_Status.py") == 1
    assert not any("trend" in d.title.lower() for d in descriptors)


def test_every_registered_page_file_exists() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    missing = [d.file_path for d in get_navigation_pages() if not (src / d.file_path).is_file()]
    assert not missing
