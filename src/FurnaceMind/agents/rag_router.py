# FurnaceMind/agents/rag_router.py
# Purpose: Query routing + field resolution + prompt injection defense
# Fixed: Concatenated token matching (eta+co → etaco), alias map for
#        operator shorthand, prevented short-token spurious matches.

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _extract_candidate_terms(user_query: str) -> List[str]:
    """
    Returns tokens + n-grams + concatenated forms so phrases like
    'eta co' also produce 'etaco' for matching against BF2_BODY_ETACO.
    """
    q = _normalize(user_query)
    toks = q.split()
    trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks) - 2)]
    bigrams = [" ".join(toks[i:i+2]) for i in range(len(toks) - 1)]

    # Concatenated forms: "eta co" → "etaco", "top pressure" → "toppressure"
    concat_bigrams = ["".join(toks[i:i+2]) for i in range(len(toks) - 1)]
    concat_trigrams = ["".join(toks[i:i+3]) for i in range(len(toks) - 2)]

    seen = set()
    out = []
    for x in (trigrams + bigrams + concat_trigrams + concat_bigrams + toks):
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ---------------------------------------------------------------------------
# Operator alias map — common shorthand → canonical terms to search for
# Add entries here as you discover how operators actually refer to parameters
# ---------------------------------------------------------------------------
OPERATOR_ALIASES: Dict[str, List[str]] = {
    "eta co":       ["etaco"],
    "eta":          ["etaco"],
    "coke rate":    ["coke_rate", "coke rate"],
    "nut coke":     ["nut_coke", "nut coke"],
    "top pressure": ["top pressure average", "top press"],
    "top temp":     ["top temp average"],
    "permeability": ["permeability"],
    "hot blast":    ["hot blast"],
    "co2":          ["co2 in bf gas"],
    "co gas":       ["co in bf gas"],
    "si":           ["silicon", "si content"],
    "hm temp":      ["hot metal temp"],
}


# ---------------------------------------------------------------------------
# Stop words — ignore during token overlap matching
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "can", "you", "me", "my", "for", "of", "in", "on", "at", "to",
    "and", "or", "it", "its", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "shall", "has", "have", "had",
    "this", "that", "these", "those", "what", "which", "who", "how",
    "show", "plot", "graph", "chart", "give", "get", "tell", "display",
    "last", "hours", "hour", "hrs", "hr", "minutes", "mins", "min",
    "okay", "ok", "cool", "please", "thanks", "trend", "data",
    "window", "avg", "average",
})

# Short tokens that are too ambiguous to match alone as substrings
_SHORT_AMBIGUOUS = frozenset({
    "co", "no", "bf", "r1", "r2", "r3", "q1", "q2", "q3", "q4",
    "bf2", "r10", "t",
})


# ---------------------------------------------------------------------------
# Field resolution (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------------
def resolve_fields_from_query(
    user_query: str,
    field_labels: Dict[str, str],
    max_fields: int = 4,
) -> List[str]:
    """
    Resolve requested fields from user text by matching against human labels.

    field_labels: { "<internal_key>": "<human_label>", ... }
    Returns: list of human-label strings that match the user's intent.

    Matching strategy (in priority order):
    1. Alias expansion — map operator shorthand to canonical search terms
    2. Exact n-gram match against normalized label keys
    3. Concatenated form match (etaco, toptemp, etc.)
    4. Substring match with minimum length requirements
    5. Multi-token scoring with ambiguity guards
    """
    if not field_labels:
        return []

    human_labels = list(field_labels.values())
    label_map = {
        _normalize(lbl): lbl
        for lbl in human_labels
        if isinstance(lbl, str) and lbl.strip()
    }

    qn = _normalize(user_query)
    candidates = _extract_candidate_terms(user_query)

    # --- Step 0: Expand aliases ---
    expanded_terms: List[str] = []
    for alias, expansions in OPERATOR_ALIASES.items():
        if alias in qn:
            expanded_terms.extend(expansions)

    # --- Pass 1: Exact n-gram / alias match against label keys ---
    hits: List[str] = []

    # Check expanded alias terms first
    for term in expanded_terms:
        tn = _normalize(term)
        for nk, orig in label_map.items():
            if tn in nk:
                hits.append(orig)

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (alias match): query={user_query!r} → {result}")
        return result

    # Check candidates (includes concatenated forms like "etaco")
    for cand in candidates:
        if cand in label_map:
            hits.append(label_map[cand])

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (exact match): query={user_query!r} → {result}")
        return result

    # --- Pass 2: Substring match (candidate inside label) ---
    # Only use candidates with length >= 4 to avoid short-token noise
    for cand in candidates:
        if len(cand) < 4:
            continue
        if cand in _SHORT_AMBIGUOUS:
            continue
        for nk, orig in label_map.items():
            if cand in nk:
                hits.append(orig)

    # Also check concatenated candidates (e.g., "etaco" inside "bf2 body etaco")
    for cand in candidates:
        if " " not in cand and len(cand) >= 5:  # concatenated forms only
            for nk, orig in label_map.items():
                if cand in nk and orig not in hits:
                    hits.append(orig)

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (substring match): query={user_query!r} → {result}")
        return result

    # --- Pass 3: Multi-token scoring ---
    query_tokens = {
        t for t in qn.split()
        if t not in _STOP_WORDS and len(t) >= 2
    }

    if not query_tokens:
        return []

    # Remove ambiguous short tokens from scoring
    safe_tokens = query_tokens - _SHORT_AMBIGUOUS

    if not safe_tokens:
        # If ALL tokens are ambiguous (e.g., "co"), still try but require higher threshold
        safe_tokens = query_tokens
        threshold = 0.8
    else:
        threshold = 0.4

    scored: List[Tuple[float, str]] = []
    for nk, orig in label_map.items():
        label_tokens = set(nk.split())

        # Exact token overlap
        overlap = safe_tokens & label_tokens
        if overlap:
            score = len(overlap) / len(safe_tokens)
            scored.append((score, orig))
            continue

        # Substring token match (e.g., "coke" found within "coke_rate")
        match_count = sum(
            1 for qt in safe_tokens
            if len(qt) >= 4 and qt in nk
        )
        if match_count > 0:
            score = match_count / len(safe_tokens)
            scored.append((score, orig))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [orig for score, orig in scored if score >= threshold]
        if hits:
            result = _unique_limit(hits, max_fields)
            logger.info(f"Field resolution (token scoring): query={user_query!r} → {result}")
            return result

    logger.info(f"Field resolution: no match for query={user_query!r}")
    return []


def _unique_limit(items: List[str], limit: int) -> List[str]:
    """Deduplicate while preserving order, then limit."""
    seen = set()
    out = []
    for h in items:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out[:limit]


# ---------------------------------------------------------------------------
# Time range + window parsing (SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------------
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
        if "minutes" in time_range:
            window = "1 minute"
        elif any(x in time_range for x in ["last 1 hours", "last 2 hours"]):
            window = "5 minutes"
        else:
            window = "15 minutes"

    return time_range, window


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
def detect_plot_intent(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["trend", "plot", "graph", "chart", "show", "curve"])


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------
def route_query(query: str, field_labels: Optional[Dict[str, str]] = None) -> str:
    """
    Routing:
    - If user mentions ANY known parameter (matches FIELD_LABELS) -> influx
    - Else if plot/live intent keywords -> influx
    - Else if shift intelligence keywords -> shift
    - Else -> knowledge
    """
    q = query.lower()

    # 1) Strongest: explicit parameter match
    if field_labels:
        hits = resolve_fields_from_query(query, field_labels)
        if hits:
            return "influx"

    # 2) Plot / live-ish intent
    plot_intent = detect_plot_intent(query)
    live_intent = any(
        k in q for k in [
            "last ", "hours", "mins", "minutes",
            "now", "current", "live", "today",
        ]
    )

    if plot_intent or live_intent:
        return "influx"

    # 3) Shift intelligence
    if any(k in q for k in [
        "shift", "fsi", "stability",
        "recurring", "influence", "anomaly",
    ]):
        return "shift"

    return "knowledge"


# ---------------------------------------------------------------------------
# Prompt injection defense
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?))"
    r"|(you\s+are\s+now\s+)"
    r"|(system\s*:\s*)"
    r"|(forget\s+(everything|all)\s+(above|before))"
    r"|(do\s+not\s+follow\s+(the|your)\s+(system|original))",
    re.IGNORECASE,
)


def sanitize_context(text: str) -> str:
    """Sanitize retrieved RAG context to defend against prompt injection."""
    if not text:
        return ""
    cleaned = _INJECTION_PATTERNS.sub("[FILTERED]", text)
    max_chars = 12_000
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n[... truncated for length ...]"
    return cleaned