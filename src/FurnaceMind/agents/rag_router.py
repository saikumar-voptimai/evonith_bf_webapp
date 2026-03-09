# FurnaceMind/agents/rag_router.py
# Purpose: Query routing + field resolution + prompt injection defense
# Fixed: Stop words applied in ALL matching passes (not just token scoring).
#        Added common non-parameter words (furnace, body, profile, etc.)
#        that appear in labels but are NOT what operators mean when they ask.

import re
import logging
from typing import List, Dict, Tuple, Optional


logger = logging.getLogger(__name__)



# Text normalization
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _extract_candidate_terms(user_query: str) -> List[str]:
    """
    Returns tokens + n-grams + concatenated forms.
    'eta co' → ['eta co', 'etaco', 'eta', 'co']
    """
    q = _normalize(user_query)
    toks = q.split()
    trigrams = [" ".join(toks[i:i+3]) for i in range(len(toks) - 2)]
    bigrams = [" ".join(toks[i:i+2]) for i in range(len(toks) - 1)]
    concat_bigrams = ["".join(toks[i:i+2]) for i in range(len(toks) - 1)]
    concat_trigrams = ["".join(toks[i:i+3]) for i in range(len(toks) - 2)]

    seen = set()
    out = []
    for x in (trigrams + bigrams + concat_trigrams + concat_bigrams + toks):
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out



# Operator alias map
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



# Stop words — IGNORED in all matching passes.
# These words are common in English queries OR appear in label names
# but are NOT themselves parameter identifiers.
_STOP_WORDS = frozenset({
    # English function words
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "can", "you", "me", "my", "for", "of", "in", "on", "at", "to",
    "and", "or", "it", "its", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "shall", "has", "have", "had",
    "this", "that", "these", "those", "what", "which", "who", "how",
    "about", "with", "from", "into", "not", "but", "just", "also",
    "very", "really", "some", "any", "all", "much", "more", "than",
    "when", "where", "why", "if", "so", "then", "now", "here",

    # Chat / command words
    "show", "plot", "graph", "chart", "give", "get", "tell", "display",
    "okay", "ok", "cool", "please", "thanks", "thank", "hey", "hi",
    "yes", "no", "yeah",

    # Time words
    "last", "hours", "hour", "hrs", "hr", "minutes", "mins", "min",
    "window", "avg", "average", "today", "yesterday", "time",

    # Data / analysis words
    "trend", "data", "value", "values", "current", "live", "recent",
    "compare", "analysis", "summary", "report",

    # Application / system words — these appear in labels but aren't parameters
    "furnace", "furnacemind", "mind", "system", "app", "application",
    "blast", "bf2", "bfbd", "bfbo", "proc",

    # Label structural words — appear in many labels but don't identify parameters
    "process", "params", "temperature", "profile", "delta",
    "body", "stave", "r10", "mm",
})


# Short tokens too ambiguous for substring matching
_SHORT_AMBIGUOUS = frozenset({
    "co", "no", "bf", "r1", "r2", "r3", "q1", "q2", "q3", "q4",
    "bf2", "r10", "t", "temp", "avg",
})


def _is_meaningful_candidate(cand: str) -> bool:
    """Check if a candidate term is meaningful (not a stop word or ambiguous)."""
    # Multi-word candidates: check if ALL words are stop words
    words = cand.split()
    if all(w in _STOP_WORDS for w in words):
        return False
    # Single-word: check directly
    if len(words) == 1 and cand in _STOP_WORDS:
        return False
    if cand in _SHORT_AMBIGUOUS:
        return False
    return True



# Field resolution (SINGLE SOURCE OF TRUTH)
def resolve_fields_from_query(
    user_query: str,
    field_labels: Dict[str, str],
    max_fields: int = 4,
) -> List[str]:
    """
    Resolve requested fields from user text by matching against human labels.

    Matching strategy (in priority order):
    1. Alias expansion
    2. Exact n-gram match
    3. Substring match (with stop word + length filtering)
    4. Multi-token scoring
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

    # Filter candidates: remove pure stop-word candidates early
    meaningful_candidates = [c for c in candidates if _is_meaningful_candidate(c)]

    # --- Step 0: Expand aliases ---
    expanded_terms: List[str] = []
    for alias, expansions in OPERATOR_ALIASES.items():
        if alias in qn:
            expanded_terms.extend(expansions)

    # --- Pass 1: Alias match ---
    hits: List[str] = []

    for term in expanded_terms:
        tn = _normalize(term)
        for nk, orig in label_map.items():
            if tn in nk:
                hits.append(orig)

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (alias): {user_query!r} → {result}")
        return result

    # --- Pass 2: Exact n-gram match ---
    for cand in meaningful_candidates:
        if cand in label_map:
            hits.append(label_map[cand])

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (exact): {user_query!r} → {result}")
        return result

    # --- Pass 3: Substring match (candidate inside label) ---
    for cand in meaningful_candidates:
        # Must be long enough AND not a stop/ambiguous word
        if len(cand) < 5:
            continue
        for nk, orig in label_map.items():
            if cand in nk:
                hits.append(orig)

    # Concatenated candidates (e.g., "etaco", "cokerate")
    for cand in meaningful_candidates:
        if " " not in cand and len(cand) >= 5:
            for nk, orig in label_map.items():
                if cand in nk and orig not in hits:
                    hits.append(orig)

    if hits:
        result = _unique_limit(hits, max_fields)
        logger.info(f"Field resolution (substring): {user_query!r} → {result}")
        return result

    # --- Pass 4: Multi-token scoring ---
    query_tokens = {
        t for t in qn.split()
        if t not in _STOP_WORDS and len(t) >= 2 and t not in _SHORT_AMBIGUOUS
    }

    if not query_tokens:
        logger.info(f"Field resolution: no meaningful tokens in {user_query!r}")
        return []

    scored: List[Tuple[float, str]] = []
    for nk, orig in label_map.items():
        label_tokens = set(nk.split())

        # Exact token overlap
        overlap = query_tokens & label_tokens
        if overlap:
            score = len(overlap) / len(query_tokens)
            scored.append((score, orig))
            continue

        # Substring token match
        match_count = sum(
            1 for qt in query_tokens
            if len(qt) >= 5 and qt in nk
        )
        if match_count > 0:
            score = match_count / len(query_tokens)
            scored.append((score, orig))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        threshold = 0.4
        hits = [orig for score, orig in scored if score >= threshold]
        if hits:
            result = _unique_limit(hits, max_fields)
            logger.info(f"Field resolution (scoring): {user_query!r} → {result}")
            return result

    logger.info(f"Field resolution: no match for {user_query!r}")
    return []


def _unique_limit(items: List[str], limit: int) -> List[str]:
    seen = set()
    out = []
    for h in items:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out[:limit]



# Time range + window parsing
def parse_time_range_and_window(user_query: str) -> Tuple[str, str]:
    q = user_query.lower()

    m = re.search(r"last\s+(\d+)\s*(h|hr|hrs|hour|hours)\b", q)
    if m:
        time_range = f"last {int(m.group(1))} hours"
    else:
        m = re.search(r"last\s+(\d+)\s*(m|min|mins|minute|minutes)\b", q)
        if m:
            time_range = f"last {int(m.group(1))} minutes"
        else:
            time_range = "last 8 hours"

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



# Intent detection
def detect_plot_intent(user_query: str) -> bool:
    q = user_query.lower()
    return any(k in q for k in ["trend", "plot", "graph", "chart", "curve"])



# Query routing
def route_query(query: str, field_labels: Optional[Dict[str, str]] = None) -> str:
    q = query.lower()

    # 1) Explicit parameter match
    if field_labels:
        hits = resolve_fields_from_query(query, field_labels)
        if hits:
            return "influx"

    # 2) Plot / live intent (only if plot-specific verbs are present)
    plot_intent = detect_plot_intent(query)
    live_intent = any(k in q for k in ["last ", "live", "now", "current"])

    if plot_intent or (live_intent and any(k in q for k in ["hour", "min", "trend"])):
        return "influx"

    # 3) Shift intelligence
    if any(k in q for k in ["shift", "fsi", "stability", "recurring", "influence", "anomaly"]):
        return "shift"

    return "knowledge"



# Prompt injection defense
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?))"
    r"|(you\s+are\s+now\s+)"
    r"|(system\s*:\s*)"
    r"|(forget\s+(everything|all)\s+(above|before))"
    r"|(do\s+not\s+follow\s+(the|your)\s+(system|original))",
    re.IGNORECASE,
)


def sanitize_context(text: str) -> str:
    if not text:
        return ""
    cleaned = _INJECTION_PATTERNS.sub("[FILTERED]", text)
    max_chars = 12_000
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n[... truncated for length ...]"
    return cleaned