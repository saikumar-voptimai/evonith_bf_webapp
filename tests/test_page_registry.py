"""Tests for shared page registry metadata and visibility."""

from src.config.page_registry import (
    HIDDEN_PAGE_PATHS,
    PAGE_REGISTRY,
    get_feedback_page_options,
    get_navigation_pages,
    is_page_visible,
)


def test_feedback_options_come_from_navigation_registry() -> None:
    """Feedback dropdown options should mirror navigation titles."""
    descriptors = get_navigation_pages()
    expected_titles = [
        descriptor.title
        for descriptor in descriptors
        if descriptor.include_in_feedback
    ]
    assert get_feedback_page_options() == expected_titles


def test_hidden_pages_stay_registered_but_not_in_navigation() -> None:
    """Hidden page implementations retain metadata without appearing in the UI."""
    registered_paths = {descriptor.file_path for descriptor in PAGE_REGISTRY}
    navigation_paths = {descriptor.file_path for descriptor in get_navigation_pages()}

    assert HIDDEN_PAGE_PATHS <= registered_paths
    assert HIDDEN_PAGE_PATHS.isdisjoint(navigation_paths)


def test_page_visibility_is_case_insensitive() -> None:
    """Welcome-page path casing must not bypass the shared visibility rule."""
    assert not is_page_visible("custom_pages/9_blend_optimizer.py")
    assert is_page_visible("custom_pages/10_scheduled_jobs.py")
