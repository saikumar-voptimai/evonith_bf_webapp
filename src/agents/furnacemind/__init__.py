"""FurnaceMind package exports.

Keep page imports lazy so non-UI modules such as ``agent`` and ``graph`` can be
imported without loading embedding providers or Streamlit page dependencies.
"""


def render_ai_cooperate(*args, **kwargs):
    """
    Render the Streamlit AI Co-Operate page.

    Args:
        - args: tuple - Positional arguments forwarded to the page renderer.
        - kwargs: dict - Keyword arguments forwarded to the page renderer.

    Returns:
        - Any
    """
    from agents.furnacemind.page import render_ai_cooperate as _render_ai_cooperate

    return _render_ai_cooperate(*args, **kwargs)


__all__ = ["render_ai_cooperate"]
