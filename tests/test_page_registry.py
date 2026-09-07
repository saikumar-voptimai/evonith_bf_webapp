"""Tests for shared page registry metadata."""

from src.config.page_registry import get_feedback_page_options, get_navigation_pages


def test_feedback_options_come_from_navigation_registry() -> None:
    """Feedback dropdown options should mirror navigation titles."""
    descriptors = get_navigation_pages()
    expected_titles = [
        descriptor.title for descriptor in descriptors if descriptor.include_in_feedback
    ]
    assert get_feedback_page_options() == expected_titles


def test_scheduled_tasks_page_is_registered() -> None:
    """Expose Scheduled Tasks through the shared application navigation."""

    descriptors = get_navigation_pages()
    assert any(
        descriptor.file_path == "custom_pages/10_Scheduled_Tasks.py"
        and descriptor.title == "Scheduled Tasks"
        for descriptor in descriptors
    )
