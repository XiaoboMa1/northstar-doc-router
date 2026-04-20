"""Classifier: keyword scoring + reconciliation decision tree.

Pure functions. No I/O, no API calls, no logging side-effects.

Public API:
    keyword_score(text, keywords, scoring_config) -> StringMatch | None
    reconcile(string_match, llm, *, schema_mismatch, missing_doc,
              batch_error_reason, config) -> Reconciled

See note/impl-spec.md §5 (reconcile), §7 (scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StringMatch:
    keyword: str
    hits: int
    score: float


@dataclass(frozen=True)
class LLMResult:
    """Subset of llm_service.LLMDocResult relevant to reconcile."""
    keyword: str
    score: float


@dataclass
class Reconciled:
    route: str
    conflict: bool
    final_confidence: float
    review_reason: Optional[str]


# ---------------------------------------------------------------------------
# Keyword scoring (impl-spec §7)
# ---------------------------------------------------------------------------


def keyword_score(
    text: str,
    keywords: list[str],
    scoring_config: dict,
) -> Optional[StringMatch]:
    """4-tier piecewise scoring per impl-spec §7:
      - hits == 0                                        → skip
      - hits >= min_hits AND density >= density_thresh   → 1.0
      - hits >= min_hits OR  density >= density_thresh   → partial_score (0.5)
      - else (single sparse hit in long doc)             → weak_score    (0.2)

    Returns highest-scoring match; ties broken by keywords list order.
    """
    if not text or not keywords:
        return None

    min_full = int(scoring_config.get("min_hits_for_full_score", 2))
    density_thresh = float(scoring_config.get("density_threshold_for_full_score", 0.02))
    partial_score = float(scoring_config.get("partial_score", 0.5))
    weak_score = float(scoring_config.get("weak_score", 0.2))

    lowered = text.lower()
    word_count = max(len(text.split()), 1)

    best: Optional[StringMatch] = None
    for kw in keywords:
        kw_low = kw.lower()
        hits = lowered.count(kw_low)
        if hits <= 0:
            continue
        density = hits / word_count
        hits_ok = hits >= min_full
        density_ok = density >= density_thresh
        if hits_ok and density_ok:
            score = 1.0
        elif hits_ok or density_ok:
            score = partial_score
        else:
            score = weak_score
        cand = StringMatch(keyword=kw_low, hits=hits, score=score)
        # keywords iterated in config order → earlier wins on strict ties.
        if best is None or cand.score > best.score:
            best = cand
    return best


# ---------------------------------------------------------------------------
# Reconciliation (impl-spec §5)
# ---------------------------------------------------------------------------


def reconcile(
    string_match: Optional[StringMatch],
    llm: Optional[LLMResult],
    *,
    schema_mismatch: bool = False,
    missing_doc: bool = False,
    batch_error_reason: Optional[str] = None,
    config: dict,
) -> Reconciled:
    """Decision tree: path A (LLM missing) | B1/B2 (disagree) | C (agree) | D (no kw).

    `batch_error_reason`: when the whole batch failed, one of
        "llm_api_error" | "llm_envelope_error" | None
    (mirrored into review_reason at Tier 0 when llm is None).

    `schema_mismatch` / `missing_doc` are pipeline-supplied booleans; they
    drive review_reason only (Tier 1). They do not alter route/confidence.
    """
    reco = config
    conflict_min = float(reco.get("conflict_keyword_min_score", 0.3))
    sm_conflict_weight = float(reco.get("string_match_conflict_weight", 0.5))
    conflict_penalty = float(reco.get("conflict_confidence_penalty", 0.8))
    fallback_penalty = float(reco.get("llm_fallback_confidence_penalty", 0.5))
    low_conf_threshold = float(reco.get("llm_low_confidence_threshold", 0.5))

    # ---- path selection ----
    if llm is None:
        # Path A: LLM failed / not sent → kw fallback
        if string_match is not None:
            route = string_match.keyword
            final_confidence = string_match.score * fallback_penalty
        else:
            route = "general"
            final_confidence = 0.0
        conflict = False
    elif string_match is not None and string_match.keyword != llm.keyword:
        # Path B: disagree. Conflict only when the two signals are comparably
        # strong: the weighted string_match must still catch up to llm.score.
        # This prevents a full-score keyword hit (e.g. "contract" in a doc
        # whose dominant theme is urgent) from dragging a confident LLM call
        # into human_review.
        weighted_sm = string_match.score * sm_conflict_weight
        if string_match.score >= conflict_min and weighted_sm >= llm.score:
            # B1: comparable strength → flag conflict; LLM wins with penalty
            route = llm.keyword
            conflict = True
            final_confidence = llm.score * conflict_penalty
        else:
            # B2: LLM decisively stronger OR string_match too weak → silent win
            route = llm.keyword
            conflict = False
            final_confidence = llm.score
    elif string_match is not None and string_match.keyword == llm.keyword:
        # Path C: agree
        route = llm.keyword
        conflict = False
        final_confidence = llm.score
    else:
        # Path D: no kw match, LLM valid
        route = llm.keyword
        conflict = False
        final_confidence = llm.score

    # ---- review_reason (priority short-circuit, impl-spec §5) ----
    review_reason: Optional[str] = None

    # Tier 0: batch-level delivery failure mirrors error_reason
    if llm is None and batch_error_reason in ("llm_api_error", "llm_envelope_error"):
        review_reason = batch_error_reason
    # Tier 1: per-doc LLM quality problems (mutually exclusive)
    elif schema_mismatch:
        review_reason = "llm_schema_mismatch"
    elif missing_doc:
        review_reason = "llm_missing_docs"
    # Tier 2: classifier signals (conflict wins low_confidence)
    elif conflict:
        review_reason = "conflict"
    elif final_confidence < low_conf_threshold:
        review_reason = "low_confidence"

    return Reconciled(
        route=route,
        conflict=conflict,
        final_confidence=round(final_confidence, 6),
        review_reason=review_reason,
    )
