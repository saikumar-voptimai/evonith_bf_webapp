"""Tests for shared page registry metadata."""

from pathlib import Path

from config.page_registry import get_feedback_page_options, get_navigation_pages


def test_feedback_options_come_from_navigation_registry() -> None:
    """Feedback dropdown options should mirror navigation titles."""
    descriptors = get_navigation_pages()
    expected_titles = [
        descriptor.title for descriptor in descriptors if descriptor.include_in_feedback
    ]
    assert get_feedback_page_options() == expected_titles


def test_furnace_status_is_only_available_inside_vboard() -> None:
    """Only one V-Board descriptor exists; Furnace Status has no standalone page."""
    descriptors = get_navigation_pages()
    assert descriptors[0].title == "Welcome"
    assert not any(d.title == "Furnace Status" for d in descriptors)
    assert not any(
        d.file_path == "custom_pages/10_Furnace_Status.py" for d in descriptors
    )
    vboard = [d for d in descriptors if d.title == "V-Board"]
    assert len(vboard) == 1
    assert vboard[0].file_path == "custom_pages/3_Data_Visualisation.py"
    assert (
        sum(d.file_path == "custom_pages/3_Data_Visualisation.py" for d in descriptors)
        == 1
    )


def test_every_registered_page_file_exists() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    missing = [
        d.file_path for d in get_navigation_pages() if not (src / d.file_path).is_file()
    ]
    assert not missing
