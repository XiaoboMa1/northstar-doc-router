"""File store: persist the four output artifacts per impl-spec §2.

Writes:
  - miscellaneous.json    bare list, route != "urgent" AND review_reason is None
  - urgent.json           bare list, route == "urgent"
  - human_review.json     bare list, review_reason is not None
  - runtime_metadata.json single object {processed_at, ended_at, model,
                                         input_file_ids, metrics}

Routing matrix (impl-spec §2):
  - miscellaneous is disjoint from both urgent and human_review.
  - urgent ∩ human_review may overlap (urgent doc with review_reason != null
    appears in BOTH, same record).

input_file_ids is assembled here (impl-spec §2.1): ordered list of doc_id in
the order records were processed, duplicates preserved. Ensures
`len(input_file_ids) == file_processed + file_errors`.

final_confidence is rounded to 6 decimals at write time (impl-spec §0) to
isolate float-noise from golden diffs. Rounding is done on a deep copy so that
the in-memory records mutated by pipeline.run stay untouched.
"""

from __future__ import annotations

import copy
import json
import os


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
    """Split into (miscellaneous, urgent, human_review) per impl-spec §2."""
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


def write_files(records: list[dict], run_metadata: dict, config: dict) -> None:
    """Write the four output files per impl-spec §2."""
    rounded = _round_final_confidence(records)
    miscellaneous, urgent, human_review = _partition(rounded)

    _write_json(config["miscellaneous_file"], miscellaneous)
    _write_json(config["urgent_file"], urgent)
    _write_json(config["human_review_file"], human_review)

    runtime_metadata = {
        "processed_at": run_metadata.get("processed_at"),
        "ended_at": run_metadata.get("ended_at"),
        "model": run_metadata.get("model"),
        "input_file_ids": [r.get("doc_id") for r in rounded],
        "metrics": run_metadata.get("metrics", {}),
    }
    _write_json(config["runtime_metadata_file"], runtime_metadata)
