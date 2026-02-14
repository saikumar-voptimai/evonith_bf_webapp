import re
from typing import List, Dict, Tuple, Optional


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _extract_candidate_terms(user_query: str) -> List[str]:
    """
    Returns tokens + n-grams so phrases like 'eta co' are detected.
    Order-preserving unique list.
    """
    q = _normalize(user_query)
    toks = q.split()
    bigrams = [" ".join(toks[i:i+2]) for i in range(len(toks) - 1)]
    trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks) - 2)]
    seen = set()
    out = []
    for x in (trigrams + bigrams + toks):
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_fields_from_query(user_query: str, field_labels: Dict[str, str], max_fields: int = 4) -> List[str]:
    """
    Resolve requested fields from user text by matching against the *human labels*
    (the values in FIELD_LABELS).

    field_labels is expected to look like:
        { "<internal_key>": "<human_label>", ... }  (as in your mcp_tools.FIELD_LABELS)

    We match against the human labels.
    Returns a list of human-label strings to plot (e.g., ["ETA_CO"] or ["Top Pressure"]).
    """
    if not field_labels:
        return []

    # Build normalized human-label lookup
    human_labels = list(field_labels.values())
    label_map = {_normalize(lbl): lbl for lbl in human_labels if isinstance(lbl, str) and lbl.strip()}

    candidates = _extract_candidate_terms(user_query)

    hits: List[str] = []
    for cand in candidates:
        if cand in label_map:
            hits.append(label_map[cand])

    # Fallback: substring match over the full normalized query
    if not hits:
        qn = _normalize(user_query)
        for nk, orig in label_map.items():
            if nk and nk in qn:
                hits.append(orig)

    # Unique + limit
    uniq = []
    seen = set()
    for h in hits:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq[:max_fields]


def parse_time_range_and_window(user_query: str) -> Tuple[str, str]:
    q = user_query.lower()

    # ---- time range ----
    m = re.search(r"last\s+(\d+)\s*(h|hr|hrs|hour|hours)\b", q)
    if m:
        n = int(m.group(1))
        time_range = f"last {n} hours"
    else:
        m = re.search(r"last\s+(\d+)\s*(m|min|mins|minute|minutes)\b", q)
        if m:
            n = int(m.group(1))
            time_range = f"last {n} minutes"
        else:
            time_range = "last 8 hours"

    # ---- averaging window ----
    m = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)\s*(avg|average)\b", q)
    if m:
        window = f"{int(m.group(1))} minutes"
    else:
        # Sensible default based on range
        if "minutes" in time_range:
            window = "1 minute"
        elif any(x in time_range for x in ["last 1 hours", "last 2 hours"]):
            window = "5 minutes"
        else:
            window = "15 minutes"

    return time_range, window


def detect_plot_intent(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["trend", "plot", "graph", "chart", "show", "curve"])


def route_query(query: str, field_labels: Optional[Dict[str, str]] = None) -> str:
    """
    Routing:
    - If user mentions ANY known parameter (matches FIELD_LABELS human labels) -> influx
    - Else if plot/live intent keywords -> influx
    - Else if shift intelligence keywords -> shift
    - Else -> knowledge
    """
    q = query.lower()

    # 1) Strongest: explicit parameter match (if we have field labels available)
    if field_labels:
        hits = resolve_fields_from_query(query, field_labels)
        if hits:
            return "influx"

    # 2) Plot / live-ish intent
    plot_intent = detect_plot_intent(query)
    live_intent = any(k in q for k in ["last ", "hours", "mins", "minutes", "now", "current", "live", "today"])

    if plot_intent or live_intent:
        return "influx"

    # 3) Shift intelligence
    if any(k in q for k in [
        "shift", "fsi", "stability",
        "recurring", "influence", "anomaly"
    ]):
        return "shift"

    return "knowledge"
