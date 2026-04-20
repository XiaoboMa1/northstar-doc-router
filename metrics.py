"""MetricsCollector: canonical per-run counters.

Field set matches note/impl-spec.md §9 and output_template.json.
`file_processed + file_errors == len(input_file_ids)`.
Incremented in-place by pipeline + llm_service; serialised into
run_metadata.metrics at end-of-run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MetricsCollector:
    # File layer (pipeline owns)
    file_processed: int = 0           # docs that passed pre-LLM gate and entered a batch
    file_errors: int = 0              # pre-LLM gate hits + file_store write failures
    total_batches: int = 0            # pack_batches output count

    # LLM layer
    llm_calls: int = 0                # API calls that returned 200 + parsed envelope
    llm_retries: int = 0              # cumulative retry attempts
    llm_api_errors: int = 0           # batches failed at API layer (retries exhausted / fatal)
    llm_tokens_used: int = 0          # sum of usage.total_tokens
    llm_schema_mismatch: int = 0      # per-doc schema violations + envelope failures (batch_size)
    llm_missing_docs: int = 0         # expected doc_ids missing from LLM response

    # Wall clock
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def show(self) -> None:
        print(
            f"file_processed={self.file_processed} "
            f"file_errors={self.file_errors} "
            f"total_batches={self.total_batches} "
            f"llm_calls={self.llm_calls} "
            f"llm_retries={self.llm_retries} "
            f"llm_api_errors={self.llm_api_errors} "
            f"llm_tokens_used={self.llm_tokens_used} "
            f"llm_schema_mismatch={self.llm_schema_mismatch} "
            f"llm_missing_docs={self.llm_missing_docs} "
            f"duration_seconds={self.duration_seconds:.2f}"
        )
