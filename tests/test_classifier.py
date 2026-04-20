"""Classifier unit tests (T9 + reconcile paths + review_reason priority).

Pure functions — no mocks, no I/O. Covers:

- T9 multi-keyword tiebreak (config-order-first wins).
- reconcile() decision tree: A (LLM missing, with/without kw), B1 (strong
  conflict), B2 (weak disagreement), C (agree), D (no kw, LLM OK).
- urgent symmetry: urgent_flag branch has been retired — urgent as a first-
  class route behaves identically to the other routes.
- review_reason priority short-circuit (tier0 mirrors batch_error_reason,
    conflict wins over low_confidence, and missing-doc uses llm_missing_docs).
"""

from __future__ import annotations

import pytest

from classifier import (
    LLMResult,
    Reconciled,
    StringMatch,
    keyword_score,
    reconcile,
)


# ---------------------------------------------------------------------------
# Config fixtures (mirror production config.json values for reconciliation
# + keyword_scoring).
# ---------------------------------------------------------------------------


RECO_CFG = {
    "conflict_keyword_min_score": 0.3,
    "string_match_conflict_weight": 0.5,
    "conflict_confidence_penalty": 0.8,
    "llm_fallback_confidence_penalty": 0.5,
    "llm_low_confidence_threshold": 0.5,
}

SCORING_CFG = {
    "min_hits_for_full_score": 2,
    "density_threshold_for_full_score": 0.02,
    "partial_score": 0.5,
    "weak_score": 0.2,
}

KEYWORDS = ["invoice", "complaint", "contract", "refund", "urgent"]


# ===========================================================================
# T9: multi-keyword tiebreak
# ===========================================================================


def test_multi_keyword_tiebreak_prefers_config_order():
    # invoice hits=2 → 1.0, complaint hits=2 → 1.0; config order says invoice
    # precedes complaint so invoice wins the tie. urgent / refund are
    # incidental signals.
    text = (
        "Urgent invoice complaint: Invoice #789 from client who filed a "
        "complaint and demands urgent refund."
    )
    sm = keyword_score(text, KEYWORDS, SCORING_CFG)
    assert sm is not None
    assert sm.keyword == "invoice"
    assert sm.score == 1.0
    # invoice literal count in the text (case-insensitive): "invoice" + "Invoice"
    assert sm.hits >= 2


def test_keyword_score_no_hits_returns_none():
    assert keyword_score("Just a random sentence.", KEYWORDS, SCORING_CFG) is None


def test_keyword_score_single_hit_yields_half_score():
    sm = keyword_score("One invoice here.", KEYWORDS, SCORING_CFG)
    assert sm is not None
    assert sm.keyword == "invoice"
    assert sm.hits == 1
    assert sm.score == 0.5


# ===========================================================================
# reconcile() — 5 decision-tree paths
# ===========================================================================


def test_reconcile_path_A_llm_none_with_kw():
    kw = StringMatch(keyword="invoice", hits=2, score=1.0)
    r = reconcile(
        kw,
        None,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "invoice"
    assert r.conflict is False
    # 1.0 * llm_fallback_confidence_penalty(0.5) = 0.5
    assert r.final_confidence == 0.5
    # 0.5 is NOT strictly < 0.5 → no low_confidence. No other trigger.
    assert r.review_reason is None


def test_reconcile_path_A_llm_none_no_kw_triggers_low_confidence():
    r = reconcile(
        None,
        None,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "general"
    assert r.conflict is False
    assert r.final_confidence == 0.0
    assert r.review_reason == "low_confidence"


def test_reconcile_path_B1_strong_conflict_llm_wins():
    # kw=urgent@1.0 vs llm=complaint@0.4.
    # weighted_sm = 1.0 * 0.5 = 0.5 >= 0.4, so B1 conflict applies.
    kw = StringMatch(keyword="urgent", hits=2, score=1.0)
    llm = LLMResult(keyword="complaint", score=0.4)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "complaint"
    assert r.conflict is True
    # 0.4 * conflict_confidence_penalty(0.8) = 0.32
    assert r.final_confidence == 0.32
    assert r.review_reason == "conflict"  # priority 1 short-circuit


def test_reconcile_path_B2_weak_disagreement_no_conflict_flag():
    # kw.score=0.2 < conflict_keyword_min_score(0.3) → B2 weak.
    kw = StringMatch(keyword="refund", hits=1, score=0.2)
    llm = LLMResult(keyword="general", score=0.9)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "general"
    assert r.conflict is False
    assert r.final_confidence == 0.9
    assert r.review_reason is None


def test_reconcile_path_C_agree():
    kw = StringMatch(keyword="invoice", hits=2, score=1.0)
    llm = LLMResult(keyword="invoice", score=0.93)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "invoice"
    assert r.conflict is False
    assert r.final_confidence == 0.93
    assert r.review_reason is None


def test_reconcile_path_D_no_kw_llm_ok():
    llm = LLMResult(keyword="general", score=0.85)
    r = reconcile(
        None,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "general"
    assert r.conflict is False
    assert r.final_confidence == 0.85
    assert r.review_reason is None


# ===========================================================================
# urgent is now a first-class route — no urgent_flag review_reason.
# ===========================================================================


def test_reconcile_urgent_symmetry_agree():
    """urgent kw + urgent llm → path C. Must behave exactly like invoice+invoice."""
    kw = StringMatch(keyword="urgent", hits=2, score=1.0)
    llm = LLMResult(keyword="urgent", score=0.9)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.route == "urgent"
    assert r.conflict is False
    assert r.final_confidence == 0.9
    assert r.review_reason is None  # urgent_flag retired — no special branch


def test_reconcile_urgent_kw_llm_says_general_no_urgent_flag():
    """Previously this would have fired review_reason="urgent_flag". Now that
    branch is gone: route is taken from LLM; if kw.score >= conflict_min it is
    'conflict' (priority 1); otherwise no review_reason from urgent alone."""
    kw = StringMatch(keyword="urgent", hits=2, score=1.0)
    llm = LLMResult(keyword="general", score=0.4)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    # weighted_sm=0.5 >= llm.score=0.4 → B1 conflict
    assert r.route == "general"
    assert r.conflict is True
    assert r.review_reason == "conflict"


# ===========================================================================
# review_reason priority: tier0 batch errors > tier1 (schema/missing) >
# tier2 (conflict > low_confidence)
# ===========================================================================


def test_review_reason_conflict_wins_over_schema_mismatch_and_low_confidence():
    # Tier 1 (schema_mismatch) outranks conflict/low_confidence in Tier 2.
    kw = StringMatch(keyword="urgent", hits=2, score=1.0)
    llm = LLMResult(keyword="complaint", score=0.1)  # low conf + conflict
    r = reconcile(
        kw,
        llm,
        schema_mismatch=True,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.conflict is True
    # 0.1 * 0.8 = 0.08 < 0.5 (low confidence would also trigger)
    assert r.final_confidence == pytest.approx(0.08, abs=1e-6)
    assert r.review_reason == "llm_schema_mismatch"


def test_review_reason_missing_doc_priority_1():
    # No conflict and no batch-level failure: missing_doc flag wins Tier 1.
    kw = StringMatch(keyword="invoice", hits=1, score=0.5)
    r = reconcile(
        kw,
        None,
        schema_mismatch=False,
        missing_doc=True,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.review_reason == "llm_missing_docs"


def test_review_reason_batch_error_mirrors_reason_tier_0():
    kw = StringMatch(keyword="invoice", hits=2, score=1.0)
    r = reconcile(
        kw,
        None,
        schema_mismatch=True,
        missing_doc=True,
        batch_error_reason="llm_envelope_error",
        config=RECO_CFG,
    )
    assert r.review_reason == "llm_envelope_error"


def test_review_reason_schema_mismatch_priority_3():
    # kw agrees with llm → no conflict; schema_mismatch=True; low conf not met.
    kw = StringMatch(keyword="invoice", hits=2, score=1.0)
    llm = LLMResult(keyword="invoice", score=0.8)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=True,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.review_reason == "llm_schema_mismatch"


def test_review_reason_low_confidence_priority_4():
    # Weak agreement, low-ish confidence, no other flags.
    kw = StringMatch(keyword="invoice", hits=2, score=1.0)
    llm = LLMResult(keyword="invoice", score=0.3)
    r = reconcile(
        kw,
        llm,
        schema_mismatch=False,
        missing_doc=False,
        batch_error_reason=None,
        config=RECO_CFG,
    )
    assert r.final_confidence == 0.3
    assert r.review_reason == "low_confidence"
