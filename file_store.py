"""File store: persist the four output artifacts per
design/implementation_specification.md §6.4 (impl-spec below).

Writes:
  - miscellaneous.json    bare list, route != "urgent" AND review_reason is None
  - urgent.json           bare list, route == "urgent"
  - human_review.json     bare list, review_reason is not None
  - runtime_metadata.json single object {processed_at, ended_at, model,
                                         input_file_ids, metrics}

Routing matrix (impl-spec §6.4):
  - miscellaneous is disjoint from both urgent and human_review.
  - urgent ∩ human_review may overlap (urgent doc with review_reason != null
    appears in BOTH, same record).

input_file_ids is assembled here: ordered list of doc_id in the order records
were processed, duplicates preserved. Ensures
`len(input_file_ids) == file_processed + file_errors`.

final_confidence is rounded to 6 decimals at write time so float noise does not
show up when two runs' output files are diffed. Rounding is done on a deep copy
so that the in-memory records mutated by pipeline.run stay untouched.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_json(path: str, payload) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _round_final_confidence(records: list[dict]) -> list[dict]:
    """Return a deep copy of records with classification.final_confidence rounded to 6 decimals."""
    out = copy.deepcopy(records)
    for r in out:
        cls = r.get("classification")
        if isinstance(cls, dict) and isinstance(cls.get("final_confidence"), (int, float)):
            cls["final_confidence"] = round(float(cls["final_confidence"]), 6)
    return out


def _partition(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into (miscellaneous, urgent, human_review) per impl-spec §6.4."""
    miscellaneous: list[dict] = []
    urgent: list[dict] = []
    human_review: list[dict] = []
    for r in records:
        cls = r.get("classification") or {}
        review_reason = cls.get("review_reason")
        is_urgent = r.get("route") == "urgent"
        needs_review = review_reason is not None
        if is_urgent:
            urgent.append(r)
        if needs_review:
            human_review.append(r)
        if not is_urgent and not needs_review:
            miscellaneous.append(r)
    return miscellaneous, urgent, human_review


def _charge_failed_deliveries(
    records: list[dict],
    charged: set[tuple],
    metrics: Any,
) -> None:
    """Move each record in a failed view file from processed to errored.

    One record can sit in two view files, so a `(doc_id, source_path)` already
    in `charged` is skipped and counts as one undelivered document.
    """
    if metrics is None:
        return
    for r in records:
        key = (r.get("doc_id"), r.get("source_path"))
        if key in charged:
            continue
        charged.add(key)
        metrics.file_errors += 1
        metrics.file_processed -= 1


def write_files(
    records: list[dict],
    run_metadata: dict,
    config: dict,
    metrics: Any = None,
) -> list[str]:
    """Write the four output files; return the paths that could not be written.

    An `OSError` on one view file does not abort the rest. Undelivered records
    are charged to `metrics.file_errors` once each, so
    `file_processed + file_errors == len(input_file_ids)` still holds in the
    metadata file after a partial write. Pass `metrics=None` to skip the
    accounting.
    """
    rounded = _round_final_confidence(records)
    miscellaneous, urgent, human_review = _partition(rounded)

    failed_paths: list[str] = []
    charged: set[tuple] = set()

    for key, payload in (
        ("miscellaneous_file", miscellaneous),
        ("urgent_file", urgent),
        ("human_review_file", human_review),
    ):
        path = config[key]
        try:
            _write_json(path, payload)
        except OSError as e:
            logger.error("output write failed: %s (%s)", path, e)
            failed_paths.append(path)
            _charge_failed_deliveries(payload, charged, metrics)

    # Read the counters after the view writes so the metadata file includes
    # the failures charged above.
    metrics_snapshot = (
        metrics.as_dict() if hasattr(metrics, "as_dict")
        else run_metadata.get("metrics", {})
    )
    runtime_metadata = {
        "processed_at": run_metadata.get("processed_at"),
        "ended_at": run_metadata.get("ended_at"),
        "model": run_metadata.get("model"),
        "input_file_ids": [r.get("doc_id") for r in rounded],
        "metrics": metrics_snapshot,
    }
    meta_path = config["runtime_metadata_file"]
    try:
        _write_json(meta_path, runtime_metadata)
    except OSError as e:
        # Run-level artifact, not a per-document delivery: no file_errors.
        logger.error("output write failed: %s (%s)", meta_path, e)
        failed_paths.append(meta_path)

    return failed_paths
